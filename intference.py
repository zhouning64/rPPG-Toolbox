""" The main function of rPPG deep learning pipeline."""

import argparse
import random

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from exp.record_video import record_video


RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--v', required=False, default="record.mpeg", type=str)
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml",
                        type=str,
                        help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    return parser


def run(config):
    model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, None)
    model_trainer.getHearRate("run_video/record.mp4", "saved_model/MMPD_MMPD_UBFC_tscan_Epoch39.pth")

if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    torch.autograd.set_detect_anomaly(True)

    record_video(
        duration=35,
        output_file="run_video/record.mp4",
        camera_index=0,
        width=240,
        height=320,
        fps=30.0
    )

    run(config)