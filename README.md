## Requirements

- `torch==1.7.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`
- `java-1.8.0`


## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Explanation of each part
models: This contains the code for network models. The two datasets require different numbers of input images, indicated by _iu_xray and _mimic_cxr suffixes.
modules: Implementation of various internal modules of the network.
pycocoevalcap: Code for evaluation metrics.
main.py: The main function. The `training_setting` allows for the selection of a specific model.
infer.py: Script for outputting generated reports.
prepare_data.py: Script for generating annotations for different regions.
XX.sh: Command execution scripts.

## preprocess

Before training, first run `prepare_data.py` to generate a structured `annotation.json` file with the reports.

