![Image of Yaktocat](image.png)


# Liverpool-Ion-Switching
In this [repository](https://github.com/stdereka/liverpool-ion-switching) you can find a outline of how to reproduce my solution for the Liverpool-Ion-Switching competition.
It contains all the code and pipelines I used to create my winning submissions.
If you run into any trouble with the setup/code or have any questions please contact me at [st.dereka@gmail.com](st.dereka@gmail.com).

## Contents

* `./preprocessing` - preprocessing scripts
* `./data` - raw and preprocessed data
* `./config` - configuration files (.json)
* `./models` - serialized copies of models and their predictions
* `./model` - train and inference pipelines of models
* `./postprocessing` - code to write submissions and do some postprocessing
* `./submissions` - final submissions

## Software requirements

* Python 3.6.9
* CUDA 10.1
* Nvidia Driver 418.67
* Python packages are detailed in `requirements.txt`. In order to install them run:
```bash
pip install -r requirements.txt
```

## Hardware requirements (recommended)

* 30 GB free disk space
* 20 GB RAM
* 1 x Tesla P100-PCIE-16GB

## Entry points

To make reproducing easier I created following scripts:

* `prepare_data.py` - reads parameters from `./config/PREPROCESSING.json` and runs preprocessing pipeline
* `train.py` - reads parameters from `./config/RFC.json` and `./config/WAVENET.json`, runs training pipelines
* `predict.py` - reads parameters from `./config/RFC.json` and `./config/WAVENET.json`, runs inference pipelines and writes submissions.


# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
# below are the shell commands used in each step, as run from the top level directory
mkdir -p data/stage1/
cd data/stage1/
kaggle competitions download -c <competition name> -f train.csv
kaggle competitions download -c <competition name> -f test_stage_1.csv

mkdir -p data/stage2/
cd ../data/stage1/
kaggle competitions download -c <competition name> -f test_stage_2.csv
cd ..

# DATA PROCESSING
# The train/predict code will also call this script if it has not already been run on the relevant data.
python ./train_code/prepare_data.py --data_dir=data/stage1/ --output_dir=data/stage1_cleaned

# MODEL BUILD: There are three options to produce the solution.
1) very fast prediction
    a) runs in a few minutes
    b) uses precomputed neural network predictions
2) ordinary prediction
    a) expect this to run for 1-2 days
    b) uses binary model files
3) retrain models
    a) expect this to run about a week
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch

shell command to run each build is below
# 1) very fast prediction (overwrites comp_preds/sub1.csv and comp_preds/sub2.csv)
python ./predict_code/calibrate_model.py

# 2) ordinary prediction (overwrites predictions in comp_preds directory)
sh ./predict_code/predict_models.sh

# 3) retrain models (overwrites models in comp_model directory)
sh ./train_code/train_models.sh
