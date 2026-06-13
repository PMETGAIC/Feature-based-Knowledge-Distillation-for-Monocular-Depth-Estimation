import os, torch, numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF
from torchvision import tv_tensors
from datasets import load_dataset

class NYUDataset(Dataset):
    def __init__(self, hf_dataset, is_train=True):
        self.dataset, self.is_train = hf_dataset, is_train
        self.spatial_ops = v2.Compose([v2.Resize(280, antialias=True), v2.RandomCrop(256), v2.RandomHorizontalFlip(p=0.5)]) if self.is_train else v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(256)])
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = tv_tensors.Image(TF.to_image(item["image"].convert("RGB")))
        depth = tv_tensors.Mask(torch.from_numpy(np.array(item["depth_map"])).unsqueeze(0).float()/10.0)
        img, depth = self.spatial_ops(img, depth)
        return self.normalize(TF.to_dtype(img, torch.float32, scale=True)), depth

class NYUDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, limit=None, fraction=1):
        super().__init__()
        self.batch_size, self.limit, self.fraction = batch_size, limit, fraction
        self.path_dati = os.path.abspath(os.path.join(os.getcwd(), "data"))
    def setup(self, stage=None):
        ds = load_dataset("sayakpaul/nyu_depth_v2", trust_remote_code=True, cache_dir=self.path_dati)
        train_ds, val_ds = ds["train"], ds["validation"]
        if self.fraction<1.0: train_ds = train_ds.shuffle(seed=42).select(range(int(len(train_ds)*self.fraction)))
        if self.limit: train_ds, val_ds = train_ds.select(range(min(self.limit, len(train_ds)))), val_ds.select(range(min(self.limit, len(val_ds))))
        self.train_ds, self.val_ds = NYUDataset(train_ds, is_train=True), NYUDataset(val_ds, is_train=False)
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)