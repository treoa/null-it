import os
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from typing import Any, Set, List, Tuple, Hashable


class MyDataset(Dataset):
    def __init__(self) -> None:
        print(f"Initializing a dataset")
        df = pd.read_csv('income_evaluation.csv')
        df.rename(columns=lambda x: x.strip(), inplace=True)
        df['income'] = df['income'].str.strip().replace('<=50K', 1).replace('>50K', 2)
        df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df_obj = df.select_dtypes(['object'])
        df_obj_cols = list(df_obj.columns)
        for i in df_obj_cols:
            unique_vals = list(df_obj[i].unique())
            for n in range(len(unique_vals)):
                df[i] = df[i].replace(unique_vals[n], n)
        xy = df.to_numpy().astype('float32')
        self.x = torch.from_numpy(xy[:, :-1])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# dataset = MyDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

# dataloader = DataLoader(dataset=dataset, batch_size = 32, shuffle=True, num_workers = 2)
# dataiter = iter(dataloader)
# data = dataiter.next()


class MyDatasets(pl.LightningDataModule):
    def __init__(self, data_dir: str = './income_evaluation.csv', batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.my_data = data_dir

    def prepare_data(self) -> None:
        self.df = pd.read_csv(f'{self.my_data}')

    def setup(self) -> None:
        self.cur_dataset = MyDataset()
        transform = transforms.Compose([transforms.ToTensor])
        first = int(len(my_dataset) * 0.8)
        self.lengths = [first, int(len(self.cur_dataset) - first)]
        self.train_data, self.val_data = random_split(self.cur_dataset, lengths=self.lengths)

    def train_dataloader(self):
        print(f"Loaded the train dataloader")
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        print(f"Loaded the validation dataloader")
        return DataLoader(self.val_data, batch_size=self.batch_size)


class LinearModel(pl.LightningModule):
    def __init__(self, dims) -> None:
        super().__init__()
        print(f"Initializing the model")
        self.l1 = nn.Linear(14, 64)
        self.l2 = nn.Linear(64, 2)
        self.do = nn.Dropout(0.16)
        self.out = nn.Sigmoid()
        self.lr = 1e-5
        self.dim = dims
        self.batch_size = 64
        self.cur_dataset = MyDataset()
        self.loss_fn = nn.CrossEntropyLoss()
        first = int(len(my_dataset) * 0.8)
        self.lengths = [first, int(len(self.cur_dataset) - first)]
        self.train_data, self.val_data = random_split(self.cur_dataset, lengths=self.lengths)
        self.save_hyperparameters()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, x) -> Any:
        batch_size, *dims = x.size()
        x = x.view(batch_size, -1)
        # print(f"\n\nThe x1 is {x.shape}\n\n\n")
        x = F.relu(self.l1(x))
        out = self.l2(x)
        # print(f"The x2 is {x.shape}\n\n\n")
        out = self.do(out)
        out = self.out(out)
        # print(f"The x3 is {x.shape}\n\n\n")
        # out = F.log_softmax(out, dim=1)
        print(f"Exiting the forward fn with shape {out.shape}")
        return out
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr = self.lr)

    def train_dataloader(self):
        print(f"Loaded the train dataloader")
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        print(f"My batch is {batch}")
        logits = self(x)
        print(f"\n\n\nI am in the training step and the shapes of logits and y are given respectively {logits.shape}\n\n {y.shape}\n\n\n")
        loss = self.loss_fn(logits, y.long())
        preds = torch.argmax(logits, 1)
        self.log('train/loss', loss, on_epoch=True)
        self.train_acc(preds, y)
        self.log('train/acc', self.train_acc, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def val_dataloader(self):
        print(f"Loaded the validation dataloader")
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.flatten(y)
        print(f"My batch is {batch}")
        print(f"I am in the validation step and the shape fo x is {x.shape}")
        logits = self(x)
        print(f"\n\n\nI am in the validation step and the shapes of logits and y are given respectively {logits.shape}\n\n {y.shape}\n\n\n")
        loss = self.loss_fn(logits, y.long())
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, y)
        self.log("valid/loss_epoch", loss)
        self.log('valid/acc_epoch', self.valid_acc)
        return logits

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}}


my_dataset = MyDataset()
first = int(len(my_dataset) * 0.8)
lengths = [first, int(len(my_dataset) - first)]
train_data, val_data = random_split(my_dataset, lengths=lengths)
train_loader = DataLoader(train_data, batch_size=64)
wandb_logger = WandbLogger()
model = LinearModel(dims = 64)
print(f"The overview of datastet: {model}")
trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate = 8, max_epochs = 20, logger = wandb_logger)
trainer.fit(model, train_loader)

