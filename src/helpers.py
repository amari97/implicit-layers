import os
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader,Dataset
import lightning.pytorch as pl 
import torch
from models import utils
from torchvision import transforms

class NoisyMNIST(MNIST):
    def __init__(self, *args,noise_const=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_const=noise_const

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img_noisy=img.detach().clone()
        
        if self.noise_const>0:
            # add N(0.5,self.noise_const^2) noise
            img_noisy=img+(torch.randn(img_noisy.shape)*self.noise_const+0.5)
        return img_noisy, img, label
    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../data/",batch_size=128,prop=0.2, noise_level_test=0.1,noise_level_train=0.0):
        super().__init__()
        self.generator = torch.Generator().manual_seed(42)
        self.prop=prop
        self.data_dir = data_dir
        self.batch_size=batch_size
        self.noise_level_test=noise_level_test
        self.noise_level_train=noise_level_train
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        NoisyMNIST(self.data_dir, train=True, download=True)
        NoisyMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = NoisyMNIST(self.data_dir, train=True,noise_const=self.noise_level_train, transform=self.transform)
            perm = torch.randperm(len(mnist_full),generator=self.generator)
            k=self.prop*len(mnist_full)
            idx = perm[:int(k)]
            mnist_full=torch.utils.data.Subset(mnist_full, idx)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [0.9, 0.1],generator=self.generator)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            mnist_test = NoisyMNIST(self.data_dir, train=False,noise_const=self.noise_level_test, transform=self.transform)
            perm = torch.randperm(len(mnist_test),generator=self.generator)
            k=self.prop*len(mnist_test)
            idx = perm[:int(k)]
            self.mnist_test=torch.utils.data.Subset(mnist_test, idx)

        if stage == "predict":
            self.mnist_predict = NoisyMNIST(self.data_dir, train=False,noise_const=0.1, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


class SwissgridDataset(Dataset):
    def __init__(self, start="2018",end="2022",train=True):
        X=utils.get_covariates(start,end)
        y = utils.load_national_consumption(start,end)/1e6
        X=X[['hours','weekday','t','2m_air_temperature']]
        col=X['2m_air_temperature']
        X['2m_air_temperature']=(col-min(col))/(max(col)-min(col))
        if train:
            X=X.loc[:"2021"]
            y=y.loc[:"2021"]
        else:
            X=X.loc["2022":]
            y=y.loc["2022":]
            
        self.target = y
        self.X = X

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.Tensor(self.X.iloc[idx].to_numpy()), torch.Tensor(self.target.iloc[idx]).view(1,1)

class SwissGridDataModule(pl.LightningDataModule):
    def __init__(self,start="2018",end="2022",batch_size=1024):
        super().__init__()
        self.start=start
        self.end=end
        self.batch_size=batch_size

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            data = SwissgridDataset(self.start,self.end, train=True)
            
            self.data_train, self.data_val = random_split(data, [int(len(data)*0.8), len(data)-int(len(data)*0.8)])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = SwissgridDataset(self.start,self.end, train=False)

        if stage == "predict":
            self.data_predict = SwissgridDataset(self.start,self.end, train=False)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)
    