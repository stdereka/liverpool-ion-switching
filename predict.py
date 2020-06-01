from model.rfc import predict_rfc
from model.wavenet import predict_wavenet
import json
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--rfc", action="store_true", help="Run rfc training pipeline")
parser.add_argument("--wavenet", action="store_true", help="Run wavenet training pipeline")
args = parser.parse_args()


if __name__ == '__main__':
    if args.rfc:
        with open('./config/RFC.json') as settings:
            configs = json.load(settings)

        print("Making test prediction with RFC model...")
        for config in tqdm(configs):
            predict_rfc(config)
        print("Done")

    if args.wavenet:
        with open('./config/WAVENET.json') as settings:
            configs = json.load(settings)

        print("Making test prediction with wavenet model...")
        for config in tqdm(configs):
            predict_wavenet(config)
        print("Done")
