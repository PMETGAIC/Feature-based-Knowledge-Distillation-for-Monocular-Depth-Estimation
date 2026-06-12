import os, argparse, random
import numpy as np
import matplotlib.pyplot as plt
import torch, timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF
from torchvision import tv_tensors
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import load_dataset

plt.switch_backend('Agg')

def verify_predictions(images, depths, preds, task_name, n=4):
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1), torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    fig, axes = plt.subplots(n, 3, figsize=(15, n*4))
    for i in range(n):
        axes[i,0].imshow((images[i].cpu()*std+mean).clamp(0,1).permute(1,2,0).numpy()); axes[i,0].axis("off")
        im_p = axes[i,1].imshow(preds[i].cpu().squeeze().numpy(), cmap="plasma_r"); axes[i,1].axis("off"); plt.colorbar(im_p, ax=axes[i,1], fraction=0.046, pad=0.04)
        im_d = axes[i,2].imshow(depths[i].cpu().squeeze().numpy(), cmap="plasma_r"); axes[i,2].axis("off"); plt.colorbar(im_d, ax=axes[i,2], fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(f"result_{task_name}.png")

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

class PaperLoss(nn.Module): # this is the combined loss from the paper 'Monocular Depth Estimation Using Multi Scale Neural Network And Feature Fusion' [https://arxiv.org/abs/2009.09934]
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
    def forward(self, pred, target, mask):
        if not mask.any(): return torch.tensor(0.0, device=pred.device, requires_grad=True)
        l1 = F.l1_loss(pred[mask], target[mask]) # MAE error
        p, t = pred*mask, target*mask
        mu_p, mu_t = F.avg_pool2d(p, 3, 1, 1), F.avg_pool2d(t, 3, 1, 1)             
        sig_p = F.avg_pool2d(p**2, 3, 1, 1) - mu_p**2
        sig_t = F.avg_pool2d(t**2, 3, 1, 1) - mu_t**2
        sig_pt = F.avg_pool2d(p*t, 3, 1, 1) - mu_p*mu_t
        ssim = ((2*mu_p*mu_t + 1e-4)*(2*sig_pt + 1e-4)) / ((mu_p**2 + mu_t**2 + 1e-4)*(sig_p + sig_t + 1e-4)) # SSIM error (Structural Similarity Index) [https://arxiv.org/abs/2006.13846]
        return self.alpha*l1 + self.beta*torch.clamp((1-ssim)/2, 0, 1)[mask].mean() # Combined Error

def relational_loss(s, t):  # loss for spatial relation between subject 
    s_pool, t_pool = F.normalize(F.adaptive_avg_pool2d(s, (16,16)).flatten(2), dim=1), F.normalize(F.adaptive_avg_pool2d(t, (16,16)).flatten(2), dim=1)
    return F.mse_loss(s_pool.transpose(1,2) @ s_pool, t_pool.transpose(1,2) @ t_pool)

def at_loss(s, t):  # attention loss
    s_map, t_map = F.interpolate(s.pow(2).mean(1, keepdim=True), size=t.shape[2:], mode='bilinear'), t.pow(2).mean(1, keepdim=True)
    return F.mse_loss(F.normalize(s_map.flatten(1), dim=1), F.normalize(t_map.flatten(1), dim=1))

def tv_loss(p): #
    return torch.mean(torch.abs(p[:,:,1:,:] - p[:,:,:-1,:])) + torch.mean(torch.abs(p[:,:,:,1:] - p[:,:,:,:-1]))

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False), nn.BatchNorm2d(out_c)
        self.conv2, self.bn2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c)
        self.skip = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)) if stride!=1 or in_c!=out_c else nn.Sequential()
    def forward(self, x): return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))

class Model_depth(nn.Module):
    def __init__(self, dim_in, dim_out, is_mini=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        ch = [16, 32, 64, 128, 256] if is_mini else [32, 64, 128, 256, 512]
        self.enc1, self.enc2, self.enc3, self.enc4 = ResBlock(dim_in, ch[0]), ResBlock(ch[0], ch[1]), ResBlock(ch[1], ch[2]), ResBlock(ch[2], ch[3])
        self.bottleneck = ResBlock(ch[3], ch[4])
        self.up1, self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[4], ch[3], 1)), ResBlock(ch[4], ch[3])
        self.up2, self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[3], ch[2], 1)), ResBlock(ch[3], ch[2])
        self.up3, self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[2], ch[1], 1)), ResBlock(ch[2], ch[1])
        self.up4, self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[1], ch[0], 1)), ResBlock(ch[1], ch[0])
        self.final = nn.Sequential(nn.Conv2d(ch[0], dim_out, 1), nn.Softplus())
    def forward(self, x, return_features=True):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        d1 = self.dec1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), s1], dim=1))
        return (self.final(d4), [s1, s2, s3, s4, b]) if return_features else self.final(d4)

class Timm_Depth(nn.Module):
    def __init__(self, dim_out=1, backbone='resnet18'):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
        ch = self.encoder.feature_info.channels()
        self.up1, self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[4], ch[3], 1)), ResBlock(ch[3]*2, ch[3])
        self.up2, self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[3], ch[2], 1)), ResBlock(ch[2]*2, ch[2])
        self.up3, self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[2], ch[1], 1)), ResBlock(ch[1]*2, ch[1])
        self.up4, self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[1], ch[0], 1)), ResBlock(ch[0]*2, ch[0])
        self.final = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[0], dim_out, 1), nn.Softplus())
    def forward(self, x, return_features=True):
        feat = self.encoder(x)
        d1 = self.dec1(torch.cat([self.up1(feat[4]), feat[3]], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), feat[2]], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), feat[1]], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), feat[0]], dim=1))
        return (self.final(d4), feat) if return_features else self.final(d4)

class BaselineTask(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model, self.lr, self.criterion = model, lr, PaperLoss()
    def forward(self, x): return self.model(x)
    def training_step(self, batch, batch_idx):
        loss = self.criterion(self(batch[0])[0] if isinstance(self(batch[0]), tuple) else self(batch[0]), batch[1], batch[1]>0)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        preds = self(batch[0])[0] if isinstance(self(batch[0]), tuple) else self(batch[0])
        mask = batch[1]>0
        mse, mae = F.mse_loss(preds[mask], batch[1][mask]), F.l1_loss(preds[mask], batch[1][mask])
        self.log_dict({"val/loss": mse, "val/mae": mae}, prog_bar=True)
        return mae
    def configure_optimizers(self): return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

class EncoderDistillationTask(pl.LightningModule):  
    def __init__(self, student, teacher, lr=1e-4, w_at=1.066, w_rel=3.261, w_out=1.443): # Thi weight combination was founded using optuna
        super().__init__()
        self.student, self.teacher, self.lr = student, teacher, lr
        self.w_at, self.w_rel, self.w_out = w_at, w_rel, w_out
        self.criterion = PaperLoss() 
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad = False
    def forward(self, x): return self.student(x)[0]
    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad(): t_preds, t_feats = self.teacher(x)
        preds, s_feats = self.student(x) 
        loss_at = sum(at_loss(s, t) for s, t in zip(s_feats, t_feats))/len(t_feats)
        loss_rel = sum(relational_loss(s, t) for s, t in zip(s_feats, t_feats))/len(t_feats)
        loss_task = self.criterion(preds, y, y>0)
        loss_out = F.l1_loss(preds, t_preds) 
        total_loss = loss_task + (self.w_at*loss_at) + (self.w_rel*loss_rel) + (self.w_out*loss_out)
        self.log_dict({"train/loss": total_loss}, prog_bar=True)
        return total_loss
    def validation_step(self, batch, batch_idx):
        preds = self.student(batch[0])[0]
        mask = batch[1]>0
        mse, mae = F.mse_loss(preds[mask], batch[1][mask]), F.l1_loss(preds[mask], batch[1][mask])
        self.log_dict({"val/loss": mse, "val/mae": mae}, prog_bar=True)
        return mae
    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=1e-2)
        return [opt], [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baseline", "kd"], default="baseline")
    parser.add_argument("--mode", choices=["train", "resume", "test", "finetune"], default="test")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--t_ckpt", type=str)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--use_custom_model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = f"./models/checkpoints/{args.task}"
    os.makedirs(save_dir, exist_ok=True)

    dm = NYUDataModule(batch_size=32, fraction=0.1 if not args.teacher else 1.0)  
    dm.setup()

    if args.task == "baseline":
        if args.use_custom_model: model = Model_depth(3, 1, is_mini=args.mini)
        else: model = Timm_Depth(backbone='resnet50') if args.teacher else Timm_Depth(backbone='mobilenetv3_small_050' if args.mini else 'resnet18')
        task = BaselineTask.load_from_checkpoint(args.ckpt, model=model) if args.mode in ["test", "finetune"] else BaselineTask(model)
    else:
        if args.use_custom_model:
            student_base = Model_depth(3, 1, is_mini=args.mini)
            t_model = Timm_Depth(backbone='resnet18' if args.t_ckpt and 'TA' in args.t_ckpt else 'resnet50')
        else:
            student_base = Timm_Depth(backbone='mobilenetv3_small_050' if args.mini else 'resnet18')
            t_model = Timm_Depth(backbone='resnet18' if args.mini else 'resnet50')
        if args.t_ckpt:
            ckpt = torch.load(args.t_ckpt, map_location="cpu")
            t_sd = {k.replace("student.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("student.")} if any(k.startswith("student.") for k in ckpt["state_dict"].keys()) else {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            t_model.load_state_dict(t_sd, strict=False)
        task = EncoderDistillationTask.load_from_checkpoint(args.ckpt, student=student_base, teacher=t_model, strict=False, lr=1e-4) if args.mode in ["test", "finetune"] else EncoderDistillationTask(student_base, t_model, lr=1e-4)

    csv_logger = CSVLogger(save_dir=save_dir, name="logs")
    checkpoint_callback = ModelCheckpoint(monitor="val/mae", mode="min", save_top_k=1, filename="best-{epoch:02d}-{val/mae:.4f}")
    early_stop_callback = EarlyStopping(monitor="val/mae", min_delta=0.00, patience=10, verbose=False, mode="min")
    
    trainer = pl.Trainer(max_epochs=args.epoch, accelerator="auto", precision="16-mixed", accumulate_grad_batches=1, gradient_clip_val=1.0, default_root_dir=save_dir, callbacks=[checkpoint_callback, early_stop_callback], logger=csv_logger)

    if args.mode in ["train", "resume", "finetune"]: trainer.fit(task, dm, ckpt_path=args.ckpt if args.mode=="resume" else None)

    trainer.validate(task, dm)
    task.eval().to(device)

    idx = random.sample(range(len(dm.val_ds)), 20)
    imgs, labels = torch.stack([dm.val_ds[i][0] for i in idx]).to(device), torch.stack([dm.val_ds[i][1] for i in idx]).to(device)

    with torch.no_grad(): preds = task(imgs)[0] if isinstance(task(imgs), tuple) else task(imgs)

    verify_predictions(imgs.cpu(), labels.cpu(), preds.cpu(), task_name=f"{args.task}_{'teacher' if args.teacher else 'student'}", n=20)

if __name__ == "__main__": main()