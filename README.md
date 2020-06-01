![Image of Yaktocat](image.png)


# Liverpool Ion Switching
In this [repository](https://github.com/stdereka/liverpool-ion-switching) you can find an outline of how to reproduce my 2nd place solution for [Liverpool Ion Switching](https://www.kaggle.com/c/liverpool-ion-switching/) competition.
It contains all the code and pipelines I used to create my winning submissions.
If you run into any trouble with the setup/code or have any questions please contact me at [st.dereka@gmail.com](st.dereka@gmail.com).

[Post](https://www.kaggle.com/c/liverpool-ion-switching/discussion/153991), which explains my approach to the challenge.

Kaggle [kernel](https://www.kaggle.com/stdereka/2nd-place-solution-preprocessing-tricks) illustrating my preprocessing and data augmentation strategies.

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
These requirements should be fulfilled if you want to retrain all models from scratch.
Running prediction with pretrained models consumes less resources - you don't even need a GPU.

* 30 GB free disk space
* 20 GB RAM
* 1 x Tesla P100-PCIE-16GB

## Entry points

To make reproducing easier I created following scripts:

* `prepare_data.py` - reads parameters from `./config/PREPROCESSING.json` and runs preprocessing pipeline
* `train.py` - reads parameters from `./config/RFC.json` and `./config/WAVENET.json`, runs training pipelines
* `predict.py` - reads parameters from `./config/RFC.json` and `./config/WAVENET.json`, runs inference pipelines and writes submissions.

## How to reproduce the results?

Follow these steps:

* Clone the repo:
```commandline
git clone https://github.com/stdereka/liverpool-ion-switching.git
cd liverpool-ion-switching
```
* Download data and pretrained models. If you are the competition host, you can skip this step - all necessary data is in the package:
```commandline
./download_data.sh
```
