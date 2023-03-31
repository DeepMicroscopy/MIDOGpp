import json
import pandas as pd
import os
import yaml


def aggregate(directory):
    maps = pd.DataFrame(columns=["train", "human breast cancer", "canine lung cancer", "canine lymphoma",
                                 "canine cutaneous mast cell tumor", "human neuroendocrine tumor",
                                 "canine soft tissue sarcoma", "human melanoma"])
    f1s = pd.DataFrame(columns=["train", "human breast cancer", "canine lung cancer", "canine lymphoma",
                                "canine cutaneous mast cell tumor", "human neuroendocrine tumor",
                                "canine soft tissue sarcoma", "human melanoma"])
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            map, f1 = {}, {}
            with open(os.path.join(directory, dir, "files", "config.yaml"), 'r') as stream:
                run = yaml.safe_load(stream)
            with open(os.path.join(directory, dir, "files", "metrics.json"), 'r') as f:
                metrics = json.load(f)
            f1['train'] = run['data']['value']['tumortypes']
            map['train'] = run['data']['value']['tumortypes']
            for tumortype in ["human breast cancer", "canine lung cancer", "canine lymphoma",
                              "canine cutaneous mast cell tumor", "human neuroendocrine tumor",
                              "canine soft tissue sarcoma", "human melanoma"]:
                f1[tumortype] = metrics['aggregates']['{}_f1'.format(tumortype)]
                map[tumortype] = metrics['aggregates']['{}_mAP'.format(tumortype)]
            f1s = f1s.append(f1, ignore_index=True)
            maps = maps.append(map, ignore_index=True)
        break

    f1s.to_csv(os.path.join(directory, "F1s.csv"), sep=";", index=False)
    maps.to_csv(os.path.join(directory, "mAPs.csv"), sep=";", index=False)

