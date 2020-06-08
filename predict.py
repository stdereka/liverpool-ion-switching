from model.rfc import predict_rfc
from model.wavenet import predict_wavenet
from postprocessing.blending import blend_predictions
from evaluation.eval import print_model_quality_report
from postprocessing.submission import save_submission, inject_residual_prediction
import json
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--rfc", action="store_true", help="Run rfc training pipeline")
parser.add_argument("--wavenet", action="store_true", help="Run wavenet training pipeline")
parser.add_argument("--blend", action="store_true", help="Blend predictions")
parser.add_argument("--eval", action="store_true", help="Show model quality report")
parser.add_argument("--sub", action="store_true", help="Write submissions")
parser.add_argument("--all", action="store_true", help="Run everything")
args = parser.parse_args()


if __name__ == '__main__':
    if args.rfc or args.all:
        with open('./config/RFC.json') as settings:
            configs = json.load(settings)

        print("Making test prediction with RFC model...")
        for config in tqdm(configs):
            predict_rfc(config)
        print("Done")

    if args.wavenet or args.all:
        with open('./config/WAVENET.json') as settings:
            configs = json.load(settings)

        print("Making test prediction with wavenet model...")
        for config in tqdm(configs):
            predict_wavenet(config)
        print("Done")

    if args.blend or args.all:
        with open('./config/BLENDING.json') as settings:
            configs = json.load(settings)

        print("Blending predictions...")
        for config in tqdm(configs):
            blend_predictions(config)
        print("Done")

    if args.eval or args.all:
        print("Evaluating model quality...")
        print_model_quality_report("models/wavenet_blend/train_proba.npy",
                                   "data/processed/synthetic/train_synthetic.csv")
        print("Done")

    if args.sub or args.all:
        print("Writing submission...")
        save_submission("wavenet_blend/test_proba.npy", "submission.csv")
        inject_residual_prediction("submission.csv", "wavenet_reduced/test_wavenet_proba.npy",
                                   "synthetic_reduced/residual.npy", "submission_injected.csv")
        print("Done")
