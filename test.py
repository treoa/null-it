import torch

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Dataset

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


dataset = MyDataset()
first_data = dataset[0]
features, labels = first_data
print(features, labels)
print(f"The 0.2 length of the dataset equals to {int(len(dataset) * 0.2)}")

dataloader = DataLoader(dataset=dataset, batch_size = 64, shuffle=True, num_workers = 2)
dataiter = iter(dataloader)
data = dataiter.next()
print(data[0].shape)