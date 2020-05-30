from preprocessing.preprocess import run_preprocessing
import json
from tqdm import tqdm


if __name__ == '__main__':
    with open('./config/PREPROCESSING.json') as settings:
        configs = json.load(settings)

    print("Building datasets...")
    for config in tqdm(configs):
        run_preprocessing(config)
    print("Done")
