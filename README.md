# TIGER

TIGER is an end-to-end deep learning framework for reconstructing aircraft trajectories from multi-station ADS-B signal observations.

## Environment Setup

Create a conda environment with Python 3.8:

```bash
conda create -n tiger python=3.8
conda activate tiger
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If you want to use GPU acceleration, install the PyTorch build that matches your CUDA version. The experiments used PyTorch 2.4.1 with CUDA 12.4.

## Dataset and Preprocessing

The dataset, processed files, and preprocessing scripts are hosted on Hugging Face:

[https://huggingface.co/datasets/LianBL/TIGER/tree/main](https://huggingface.co/datasets/LianBL/TIGER/tree/main)

You can either download the files from the web page manually, or use the Hugging Face CLI:

```bash
pip install -U huggingface_hub
huggingface-cli download LianBL/TIGER --repo-type dataset --local-dir ./TIGER-data
```

The repository contains the dataset files and the preprocessing scripts used to generate the model-ready training data. For running `tiger-v1.py` or `tiger-v2.py` directly, use the processed files and place them in an `output/` directory next to the training scripts:

```text
output/
  Train_Features.pkl
  Train_Labels.pkl
  Test_Features.pkl
  Test_Labels.pkl
  Test_Features_final.pkl
  Test_Labels_final.pkl
  scaler_feat_min.npy
  scaler_feat_max.npy
  scaler_lab_min.npy
  scaler_lab_max.npy
  count_single_test.pkl
  count_all_test.pkl
```

The training scripts load these files from `output/` at runtime. If your downloaded files are stored in another directory, copy or move the processed files into `output/` before training.

If you want to regenerate the processed files from the raw data, use the preprocessing scripts provided in the Hugging Face dataset repository. The preprocessing pipeline is responsible for sorting the raw records, selecting sensor measurements, building sliding-window samples, splitting train/test flights, and saving the scaler files used for normalization.

The expected workflow is:

```text
raw dataset
  -> preprocessing scripts
  -> output/*.pkl and output/*.npy
  -> tiger-v1.py / tiger-v2.py
```

## Training

Run TIGER V1:

```bash
python tiger-v1.py
```

Run TIGER V2:

```bash
python tiger-v2.py
```

TIGER V1 uses an Inception + BiLSTM encoder and a GRU decoder. TIGER V2 adds the SRFA attention module to the decoder.


