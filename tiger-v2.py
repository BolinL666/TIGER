import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import sys, pickle, importlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


sys.modules['numpy._core'] = importlib.import_module('numpy.core')

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as f:
        return CompatUnpickler(f).load()

import numpy as np

file_name = os.path.splitext(os.path.basename(__file__))[0]

def set_random_seed(seed_value=16396535464360928111):
    torch.manual_seed(seed_value)
    random.seed(seed_value)


label_lat_min, label_lat_max = None, None
label_lon_min, label_lon_max = None, None
label_alt_min, label_alt_max = None, None


feat_min = np.load('output/scaler_feat_min.npy')
feat_max = np.load('output/scaler_feat_max.npy')
lab_min  = np.load('output/scaler_lab_min.npy')
lab_max  = np.load('output/scaler_lab_max.npy')

label_lat_min, label_lat_max = lab_min[0], lab_max[0]
label_lon_min, label_lon_max = lab_min[1], lab_max[1]
label_alt_min, label_alt_max = lab_min[2], lab_max[2]


def load_data():
    train_X = load_pickle('output/Train_Features.pkl')
    train_Y = load_pickle('output/Train_Labels.pkl')
    test_X  = load_pickle('output/Test_Features.pkl')
    test_Y  = load_pickle('output/Test_Labels.pkl')


    return train_X, train_Y, test_X, test_Y

def load_test_data():


    test_feats = load_pickle('output/Test_Features_final.pkl')
    test_labels = load_pickle('output/Test_Labels_final.pkl')

    return test_feats, test_labels

class FlightTrajectoryDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):

        return torch.tensor(self.feats[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def load_count_data():


    count_single_test_path = os.path.join('output', 'count_single_test.pkl')
    count_all_test_path = os.path.join('output', 'count_all_test.pkl')


    with open(count_single_test_path, 'rb') as f:
        count_single_test = pickle.load(f)


    with open(count_all_test_path, 'rb') as f:
        count_all_test = pickle.load(f)


    return count_single_test, count_all_test


def normalize_features_np(x):


    return (x - feat_min) / (feat_max - feat_min)

def normalize_labels_np(y):


    return (y - lab_min) / (lab_max - lab_min)


def denormalize_labels_np(y_norm, label_lat_min, label_lat_max, label_lon_min, label_lon_max, label_alt_min, label_alt_max):


    assert y_norm.shape[-1] == 3, f"Expected last dimension size to be 3, but got {y_norm.shape[-1]}"


    lat = y_norm[:, :, 0] * (label_lat_max - label_lat_min) + label_lat_min
    lon = y_norm[:, :, 1] * (label_lon_max - label_lon_min) + label_lon_min
    alt = y_norm[:, :, 2] * (label_alt_max - label_alt_min) + label_alt_min


    return np.stack([lat, lon, alt], axis=-1)

def denormalize_labels_2d(y_norm, label_lat_min, label_lat_max, label_lon_min, label_lon_max, label_alt_min, label_alt_max):


    assert y_norm.shape[1] == 3, f"Expected second dimension size to be 3, but got {y_norm.shape[1]}"


    lat = y_norm[:, 0] * (label_lat_max - label_lat_min) + label_lat_min
    lon = y_norm[:, 1] * (label_lon_max - label_lon_min) + label_lon_min
    alt = y_norm[:, 2] * (label_alt_max - label_alt_min) + label_alt_min


    return np.stack([lat, lon, alt], axis=-1)

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    mask: np.ndarray=None) -> dict:


    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    return {
        'MAE_per_channel': mae,
        'RMSE_per_channel': rmse,
        'MAE_overall': np.mean(np.abs(err)),
        'RMSE_overall': np.sqrt(np.mean(err**2))
    }
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.branch_a = nn.Sequential(
            nn.Conv1d(15, 64, kernel_size=1, stride=3, padding=1),
            nn.ReLU()
        )
        self.branch_b = nn.Sequential(
            nn.Conv1d(15, 64, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=3, padding=1),
            nn.ReLU()
        )
        self.branch_c = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=3, padding=1),
            nn.Conv1d(15, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch_d = nn.Sequential(
            nn.Conv1d(15, 64, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=3, padding=1),
            nn.ReLU()
        )

        self.encoder_lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_init = nn.Linear(256, 128)
        self.attn_query = nn.Linear(128, 256)
        self.decoder_cell = nn.GRUCell(input_size=3, hidden_size=128)
        self.project = nn.Linear(in_features=384, out_features=3)

    def forward(self, x: torch.Tensor, target: torch.Tensor = None, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        batch_size = x.size(0)
        out_a = self.branch_a(x)
        out_b = self.branch_b(x)
        out_c = self.branch_c(x)
        out_d = self.branch_d(x)
        encoder_in = torch.cat((out_a, out_b, out_c, out_d), dim=1)
        encoder_in = encoder_in.permute(0, 2, 1).contiguous()
        encoder_seq, (h_n, _) = self.encoder_lstm(encoder_in)
        encoder_seq = self.dropout(encoder_seq)
        ctx = torch.cat((h_n[-2], h_n[-1]), dim=1)
        dec_hidden = torch.tanh(self.hidden_init(ctx))
        seq_len_out = 64
        outputs = torch.zeros(batch_size, seq_len_out, 3, device=x.device)
        dec_input = torch.zeros(batch_size, 3, device=x.device)

        for t in range(seq_len_out):
            dec_hidden = self.decoder_cell(dec_input, dec_hidden)
            query = self.attn_query(dec_hidden)
            attn_scores = torch.bmm(encoder_seq, query.unsqueeze(2)).squeeze(2)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_seq).squeeze(1)
            h_fusion = torch.cat((dec_hidden, context), dim=1)
            pred_t = self.project(h_fusion)
            outputs[:, t, :] = pred_t
            if (target is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                dec_input = target[:, t, :]
            else:
                dec_input = pred_t

        return outputs


def mse_loss(pred, target, weights=(1.0, 1.0, 1.0)):


    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} target {target.shape}"


    loss = (pred - target) ** 2


    loss_lat = (loss[:, :, 0].sum() / loss[:, :, 0].numel()) * weights[0]
    loss_lon = (loss[:, :, 1].sum() / loss[:, :, 1].numel()) * weights[1]
    loss_alt = (loss[:, :, 2].sum() / loss[:, :, 2].numel()) * weights[2]


    computed_loss = loss_lat + loss_lon + loss_alt
    return computed_loss

def find_next_run_number(base_dir):
    run_numbers = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path) and d.startswith('run_'):
            try:
                run_num = int(d.split('_')[1])
                run_numbers.append(run_num)
            except ValueError:
                continue
    if not run_numbers:
        return 1
    next_run_number = max(run_numbers) + 1
    return next_run_number


def latlonalt_to_ecef(lat, lon, alt):

    a = 6378.137
    b = 6356.752
    e2 = 1 - (b**2 / a**2)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)


    RN = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)


    x = (RN + alt*1e-3) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (RN + alt*1e-3) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (RN * (1 - e2) + alt*1e-3) * np.sin(lat_rad)

    return x, y, z

def calculate_mde(true_values, pred_values):


    true_ecef = np.array([latlonalt_to_ecef(lat, lon, alt) for lat, lon, alt in true_values])
    pred_ecef = np.array([latlonalt_to_ecef(lat, lon, alt) for lat, lon, alt in pred_values])


    error_squared = np.sum((true_ecef - pred_ecef)**2, axis=1)
    mse_per_trajectory = np.mean(error_squared, axis=0)
    rmse_per_trajectory = np.sqrt(mse_per_trajectory)


    mde = np.mean(rmse_per_trajectory)

    return mde

def compute_bearing(lat1_deg, lon1_deg, lat2_deg, lon2_deg):


    lat1 = np.radians(lat1_deg)
    lat2 = np.radians(lat2_deg)
    dlon = np.radians(lon2_deg - lon1_deg)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    mag = np.sqrt(x * x + y * y)
    if mag < 1e-12:
        return np.nan
    return np.degrees(np.arctan2(x, y))

def compute_heading_deviation(all_valid_gt_list, all_valid_pred_list):


    errors = []
    for gt, pred in zip(all_valid_gt_list, all_valid_pred_list):
        if len(gt) < 2:
            continue

        h_true = compute_bearing(gt[0, 0],  gt[0, 1],
                                 gt[-1, 0], gt[-1, 1])

        h_pred = compute_bearing(pred[0, 0],  pred[0, 1],
                                 pred[-1, 0], pred[-1, 1])
        if np.isnan(h_true) or np.isnan(h_pred):
            continue
        diff = h_pred - h_true
        diff = (diff + 180.0) % 360.0 - 180.0
        errors.append(abs(diff))
    if not errors:
        return float('nan')
    return float(np.mean(errors))


def compute_final_position_error(all_valid_gt_list, all_valid_pred_list):


    errors = []
    for gt, pred in zip(all_valid_gt_list, all_valid_pred_list):
        if len(gt) == 0:
            continue
        x_t, y_t, z_t = latlonalt_to_ecef(gt[-1, 0],   gt[-1, 1],   gt[-1, 2])
        x_p, y_p, z_p = latlonalt_to_ecef(pred[-1, 0], pred[-1, 1], pred[-1, 2])
        dist_km = np.sqrt((x_t - x_p) ** 2 +
                          (y_t - y_p) ** 2 +
                          (z_t - z_p) ** 2)
        errors.append(dist_km)
    if not errors:
        return float('nan')
    return float(np.mean(errors))

def compute_frechet_distance(all_valid_gt_list, all_valid_pred_list):


    fd_list = []

    for gt, pred in zip(all_valid_gt_list, all_valid_pred_list):
        T_gt   = len(gt)
        T_pred = len(pred)
        if T_gt == 0 or T_pred == 0:
            continue


        gt_ecef   = np.array([latlonalt_to_ecef(p[0], p[1], p[2]) for p in gt])
        pred_ecef = np.array([latlonalt_to_ecef(p[0], p[1], p[2]) for p in pred])


        diff = gt_ecef[:, np.newaxis, :] - pred_ecef[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))


        dp = np.full((T_gt, T_pred), np.inf)
        dp[0, 0] = dist[0, 0]
        for i in range(1, T_gt):
            dp[i, 0] = max(dp[i-1, 0], dist[i, 0])
        for j in range(1, T_pred):
            dp[0, j] = max(dp[0, j-1], dist[0, j])
        for i in range(1, T_gt):
            for j in range(1, T_pred):
                dp[i, j] = max(dist[i, j],
                               min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]))

        fd_list.append(dp[T_gt - 1, T_pred - 1])

    return float(np.mean(fd_list)) if fd_list else float('nan')


def compute_global_metrics(all_valid_gt_list, all_valid_pred_list):


    if len(all_valid_gt_list) == 0 or not all_valid_gt_list[0].size:
        print("No valid predictions to compute aggregated metrics.")
        return


    all_gt   = np.concatenate(all_valid_gt_list, axis=0)
    all_pred = np.concatenate(all_valid_pred_list, axis=0)


    lat_err = all_pred[:, 0] - all_gt[:, 0]
    lon_err = all_pred[:, 1] - all_gt[:, 1]
    alt_err = (all_pred[:, 2] - all_gt[:, 2]) / 1000.0


    lat_mae  = np.mean(np.abs(lat_err))
    lon_mae  = np.mean(np.abs(lon_err))
    alt_mae  = np.mean(np.abs(alt_err))
    lat_rmse = np.sqrt(np.mean(lat_err ** 2))
    lon_rmse = np.sqrt(np.mean(lon_err ** 2))
    alt_rmse = np.sqrt(np.mean(alt_err ** 2))


    lat_range = lab_max[0] - lab_min[0]
    lon_range = lab_max[1] - lab_min[1]
    alt_range = (lab_max[2] - lab_min[2]) / 1000.0


    lat_mre = lat_mae / lat_range * 100
    lon_mre = lon_mae / lon_range * 100
    alt_mre = alt_mae / alt_range * 100


    total_mde = calculate_mde(all_gt, all_pred)


    print("Aggregated Metrics for Test Data:")
    print(f"  MAE: Latitude: {lat_mae:.4f} deg, "
          f"Longitude: {lon_mae:.4f} deg, Altitude: {alt_mae:.4f} km")
    print(f"  RMSE: Latitude: {lat_rmse:.4f} deg, "
          f"Longitude: {lon_rmse:.4f} deg, Altitude: {alt_rmse:.4f} km")
    print(f"  MRE: Latitude: {lat_mre:.2f}%, "
          f"Longitude: {lon_mre:.2f}%, Altitude: {alt_mre:.2f}%")
    print(f"  MDE (Combined): {total_mde:.4f} km")


    fpe = compute_final_position_error(all_valid_gt_list, all_valid_pred_list)
    print(f"  Final Position Error (FPE): {fpe:.4f} km")


    heading_dev = compute_heading_deviation(all_valid_gt_list, all_valid_pred_list)
    print(f"  Heading Deviation: {heading_dev:.2f} deg")


    fd = compute_frechet_distance(all_valid_gt_list, all_valid_pred_list)
    print(f"  Frechet Distance: {fd:.4f} km")


def process_overlapping_averages(slices, window_size=64):


    if not slices:
        return np.zeros((0, 3), dtype=np.float64)

    num_slices = len(slices)
    total_time_points = num_slices + window_size - 1
    summed = np.zeros((total_time_points, 3), dtype=np.float64)
    counts = np.zeros((total_time_points,), dtype=np.int32)


    for s, slice_arr in enumerate(slices):
        for k in range(window_size):
            t = s + k
            if t < total_time_points:
                summed[t] += slice_arr[k]
                counts[t] += 1


    averaged = summed / counts[:, None]

    return averaged

def train(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0


    with tqdm(train_loader, desc=f"Training Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
        for Xb_np, Yb_np in tepoch:
            optimizer.zero_grad()


            Xb = torch.tensor(
                normalize_features_np(Xb_np),
                dtype=torch.float32,
                device=device
            ).clone().detach().permute(0, 2, 1)


            Yb = torch.tensor(
                normalize_labels_np(Yb_np),
                dtype=torch.float32,
                device=device
            ).clone().detach()


            preds = model(Xb, target=Yb, teacher_forcing_ratio=0.25)


            loss = criterion(preds, Yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            tepoch.set_postfix(loss=total_loss / (tepoch.n + 1))

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Training Loss: {avg_loss:.4f}")
    return avg_loss


def test(model, all_feats, all_labels,
         criterion, device, epoch, num_epochs,
         label_lat_min, label_lat_max, label_lon_min, label_lon_max, label_alt_min, label_alt_max,
         count_single_test, count_all_test):


    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels_file = []


    with torch.no_grad():
        with tqdm(range(len(all_feats)), desc=f"Evaluating Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
            for i in tepoch:

                Xb_np = all_feats[i]
                Yb_np = all_labels[i]


                Xb = torch.tensor(
                    normalize_features_np(Xb_np),
                    dtype=torch.float32,
                    device=device
                ).clone().detach().permute(0, 2, 1)


                Yb = torch.tensor(
                    normalize_labels_np(Yb_np),
                    dtype=torch.float32,
                    device=device
                ).clone().detach()


                preds = model(Xb, target=None, teacher_forcing_ratio=0.0)


                loss = criterion(preds, Yb)
                total_loss += loss.item()


                all_preds.append(preds.cpu().numpy())
                all_labels_file.append(Yb.cpu().numpy())


                tepoch.set_postfix(loss=total_loss / (tepoch.n + 1))


    all_preds = np.concatenate(all_preds, axis=0)
    all_labels_file = np.concatenate(all_labels_file, axis=0)


    all_preds_denorm = denormalize_labels_np(
        all_preds,
        label_lat_min, label_lat_max,
        label_lon_min, label_lon_max,
        label_alt_min, label_alt_max
    )

    all_labels_denorm = denormalize_labels_np(
        all_labels_file,
        label_lat_min, label_lat_max,
        label_lon_min, label_lon_max,
        label_alt_min, label_alt_max
    )

    avg_test_loss = total_loss / len(all_feats)
    print(f"Epoch [{epoch+1}/{num_epochs}], Testing Loss: {avg_test_loss:.4f}")


    aggregated_preds = []
    aggregated_gts   = []

    pointer = 0
    for (flight_id, sample_num) in count_single_test:

        start_idx = pointer
        end_idx = pointer + sample_num


        windows_pred = [ all_preds_denorm[start_idx + j] for j in range(sample_num) ]
        windows_gt   = [ all_labels_denorm[start_idx + j] for j in range(sample_num) ]


        full_pred = process_overlapping_averages(windows_pred, window_size=64)
        full_gt   = process_overlapping_averages(windows_gt,   window_size=64)

        aggregated_preds.append(full_pred)
        aggregated_gts.append(full_gt)

        pointer = end_idx


    visualize_prediction_full(aggregated_preds, aggregated_gts,
                              count_single_test,
                              epoch=epoch+1,
                              base_dir='.',
                              device=device,
                              mode='test')


    compute_global_metrics(aggregated_gts, aggregated_preds)

    return avg_test_loss


def save_loss_convergence_plot(history, output_dir):


    os.makedirs(output_dir, exist_ok=True)
    history_df = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, "loss_history.csv")
    fig_path = os.path.join(output_dir, "loss_convergence.png")

    history_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        linewidth=2,
        markersize=7,
        label="Train Loss"
    )
    ax.plot(
        history_df["epoch"],
        history_df["test_loss"],
        marker="s",
        linewidth=2,
        markersize=7,
        label="Test Loss"
    )
    ax.set_title("Train / Test Loss Convergence", fontsize=18, pad=10)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)
    plt.close(fig)

    return fig_path, csv_path


def visualize_prediction_full(aggregated_preds, aggregated_gts, count_single_test, epoch=None, base_dir='.', device='cpu', mode='test'):


    num_flights = len(count_single_test)


    for idx in range(num_flights):

        flight_id = count_single_test[idx][0]


        real_sequence = aggregated_gts[idx]
        pred_sequence = aggregated_preds[idx]


        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')


        all_lons = np.concatenate((real_sequence[:, 1], pred_sequence[:, 1]))
        all_lats = np.concatenate((real_sequence[:, 0], pred_sequence[:, 0]))
        all_alts = np.concatenate((real_sequence[:, 2], pred_sequence[:, 2]))

        min_lon, max_lon = all_lons.min(), all_lons.max()
        min_lat, max_lat = all_lats.min(), all_lats.max()
        min_alt, max_alt = all_alts.min(), all_alts.max()


        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        alt_range = max_alt - min_alt

        margin_lon = lon_range * 0.05
        margin_lat = lat_range * 0.05
        margin_alt = alt_range * 0.05

        ax_min_lon, ax_max_lon = min_lon - margin_lon, max_lon + margin_lon
        ax_min_lat, ax_max_lat = min_lat - margin_lat, max_lat + margin_lat
        ax_min_alt, ax_max_alt = min_alt - margin_alt, max_alt + margin_alt


        ax.plot(
            real_sequence[:, 1],
            real_sequence[:, 0],
            real_sequence[:, 2],
            c='blue', label='True Trajectory', marker='o'
        )


        ax.plot(
            pred_sequence[:, 1],
            pred_sequence[:, 0],
            pred_sequence[:, 2],
            c='red', label='Predicted Trajectory', marker='x'
        )


        ax.set_xlim(ax_min_lon, ax_max_lon)
        ax.set_ylim(ax_min_lat, ax_max_lat)
        ax.set_zlim(ax_min_alt, ax_max_alt)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude")
        ax.set_title(f"Flight {flight_id} Trajectory Prediction (3D, {mode.capitalize()} Data)")
        ax.legend()


        current_filename = os.path.basename(__file__).split('.')[0]
        current_dir = os.path.dirname(os.path.abspath(__file__))

        run_dir = os.path.join(current_dir, 'runs_decoder-gru-tf0.25')
        current_file_folder = os.path.join(run_dir, current_filename)
        epoch_dir = os.path.join(current_file_folder, f"epoch_{epoch}")
        test_dir = os.path.join(epoch_dir, "test")
        os.makedirs(test_dir, exist_ok=True)

        output_file_path = os.path.join(test_dir, f"flight_{flight_id}_trajectory_{mode}.png")
        plt.savefig(output_file_path)
        plt.close(fig)


    return
if __name__ == "__main__":
    set_random_seed()


    train_X, train_Y, test_X, test_Y  = load_data()
    count_single_test, count_all_test = load_count_data()
    all_feats, all_labels = load_test_data()

    train_dataset = FlightTrajectoryDataset(train_X, train_Y)


    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    weights = (1.0, 1.0, 1.0)
    criterion = lambda p, t: mse_loss(p, t, weights=weights)


    num_epochs = 30
    current_filename = os.path.basename(__file__).split('.')[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    loss_output_dir = os.path.join(
        current_dir,
        'runs_decoder-gru-tf0.25',
        current_filename
    )
    loss_history = []

    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        test_loss = test(model, all_feats, all_labels,
                         criterion, device, epoch, num_epochs,
                         label_lat_min, label_lat_max, label_lon_min, label_lon_max, label_alt_min, label_alt_max,
                         count_single_test, count_all_test)
        loss_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss
        })
        scheduler.step()
        fig_path, csv_path = save_loss_convergence_plot(loss_history, loss_output_dir)
        print(f"Loss convergence figure saved to: {fig_path}")
        print(f"Loss history CSV saved to: {csv_path}")
