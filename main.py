import sys
from training import training
from inference import inference
from evaluation import evaluate
from aggregate import aggregate
from hydra import compose, initialize


if __name__ == '__main__':
    initialize(version_base=None, config_path="configs/")
    cfg = compose(config_name="all")
    #training(cfg)
    inference("wandb")
    evaluate("wandb")
    aggregate("wandb")
