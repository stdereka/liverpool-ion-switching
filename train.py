from model.rfc import train_rfc
import json
from tqdm import tqdm


if __name__ == '__main__':
    with open('./config/RFC.json') as settings:
        configs = json.load(settings)

    print("Training RFC models...")
    for config in tqdm(configs):
        train_rfc(config)
    print("Done")
