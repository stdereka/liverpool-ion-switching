from model.rfc import predict_rfc
import json
from tqdm import tqdm


if __name__ == '__main__':
    with open('./config/RFC.json') as settings:
        configs = json.load(settings)

    print("Making test prediction with RFC model...")
    for config in tqdm(configs):
        predict_rfc(config)
    print("Done")
