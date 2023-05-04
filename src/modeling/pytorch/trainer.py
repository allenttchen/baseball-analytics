import torch
from torch import nn, optim
from tqdm import tqdm

import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, device=None):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
        self.criterion = nn.NLLLoss() if criterion is None else criterion
        self.device = "cpu" if device is None else device
        self.model = self.model.to(self.device)

    def train(self, epochs, train_dataloader, val_dataloader=None):

        results = {
            "train_loss": [],
            "train_acc": [],
        }
        if val_dataloader:
            results["val_loss"] = []
            results["val_acc"] = []

        # Iterate over epochs
        for epoch in tqdm(range(epochs)):
            # Training the epoch
            train_loss, train_acc = self._train_epoch(
                dataloader=train_dataloader,
            )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)

            # eval the epoch
            if val_dataloader:
                val_loss, val_acc = self._eval_epoch(
                    dataloader=val_dataloader,
                )
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)
        return results

    def _train_epoch(self, dataloader):
        # Training mode
        self.model.train()

        # Iterate through batches
        total_samples = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # train steps
            probs = self.model(x)
            loss = self.criterion(probs, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # compute metrics
            num_in_batch = len(x)
            total_samples += num_in_batch
            epoch_loss += loss.detach().item() * num_in_batch
            #probs = F.softmax(logits, dim=1)
            preds = torch.max(probs, dim=1).indices
            acc = torch.sum(torch.eq(preds, y)).detach().item()
            epoch_acc += acc

        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_acc = epoch_acc / total_samples
        return avg_epoch_loss, avg_epoch_acc

    def _eval_epoch(self, dataloader):

        # Eval mode
        self.model.eval()

        # Iterate through batches
        total_samples = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                probs = self.model(x)
                loss = self.criterion(probs, y)

                # compute metrics
                num_in_batch = len(x)
                total_samples += num_in_batch
                epoch_loss += loss.detach().item() * num_in_batch
                #probs = F.softmax(logits, dim=1)
                preds = torch.max(probs, dim=1).indices
                acc = torch.sum(torch.eq(preds, y)).detach().item()
                epoch_acc += acc

        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_acc = epoch_acc / total_samples
        return avg_epoch_loss, avg_epoch_acc
