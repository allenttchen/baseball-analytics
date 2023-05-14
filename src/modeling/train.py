import os
import argparse
import json

import torch
from torch import optim, nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import set_device, read_params
from pytorch.dataset import PreprocessedDataset
from pytorch.trainer import Trainer
from models.mlp import MLP


def split(config_filepath: str):
    """Split a dataset into Train and Val"""
    config = read_params(config_filepath)
    splitting_config = config["splitting"]
    base_config = config["base"]

    complete_dataset = pd.read_csv(splitting_config["dataset_filepath"])
    train_df, val_df = train_test_split(
        complete_dataset,
        test_size=splitting_config["test_size"],
        random_state=base_config["random_state"],
    )
    train_df.to_csv(splitting_config["train_filepath"], index=False)
    val_df.to_csv(splitting_config["val_filepath"], index=False)


def train(config_filepath: str):
    """Train a model with simple routine"""

    # Load configs
    config = read_params(config_filepath)
    splitting_config = config["splitting"]
    training_config = config["training"]
    saving_config = config["saving"]

    # Config pytorch elements for model training
    device = set_device(mps=True)
    train_dataset = PreprocessedDataset(splitting_config["train_filepath"], target_name="events")
    val_dataset = PreprocessedDataset(splitting_config["val_filepath"], target_name="events")

    train_dataloader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=training_config["batch_size"], shuffle=True)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])
    criterion = nn.NLLLoss()
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

    # Model Training
    results = trainer.train(
        epochs=training_config["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    # Save results
    with open(saving_config["saved_metrics_filepath"], "w") as f:
        json.dump(results, f, indent=4)
    torch.save(trainer.model.state_dict(), saving_config["saved_model_filepath"])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config_002.yaml")
    parsed_args = args.parse_args()
    split(config_filepath=parsed_args.config)
    train(config_filepath=parsed_args.config)
