"""
import torch, argparse, random, numpy as np
import matplotlib.pyplot as plt
from Train_finale import NYUDataModule, Timm_Depth, Model_depth, BaselineTask, EncoderDistillationTask

def denormalize(t):
    return (t.cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)).clamp(0,1).permute(1,2,0).numpy()

def save_comparison(imgs, depths, preds_dict, n=10, filename="confronto_7_modelli.png"):
    n = min(n, len(imgs))
    cols = ['Input RGB'] + list(preds_dict.keys()) + ['GT']
    fig, axes = plt.subplots(n, len(cols), figsize=(3 * len(cols), n * 3))
    for i in range(n):
        axes[i, 0].imshow(denormalize(imgs[i])); axes[i, 0].axis('off')
        for j, (name, preds) in enumerate(preds_dict.items()):
            axes[i, j+1].imshow(preds[i].cpu().squeeze(), cmap='plasma'); axes[i, j+1].axis('off')
        axes[i, -1].imshow(depths[i].cpu().squeeze(), cmap='plasma'); axes[i, -1].axis('off')
        if i == 0:
            for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t_grande", type=str)
    p.add_argument("--med_base", type=str)
    p.add_argument("--med_kd", type=str)
    p.add_argument("--mini_base1", type=str)
    p.add_argument("--mini_kd1", type=str)
    p.add_argument("--mini_base2", type=str)
    p.add_argument("--mini_kd2", type=str)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = NYUDataModule(batch_size=8, fraction=.1); dm.setup()
    
    models = {}
    def load_b(ckpt, m): return BaselineTask.load_from_checkpoint(ckpt, model=m, strict=False).to(device).eval() if ckpt else None
    def load_k(ckpt, s, t): return EncoderDistillationTask.load_from_checkpoint(ckpt, student=s, teacher=t, strict=False).to(device).eval() if ckpt else None

    if args.t_grande: models["T_Grande (R50)"] = load_b(args.t_grande, Timm_Depth(backbone='resnet50'))
    if args.med_base: models["Med Base (R18)"] = load_b(args.med_base, Timm_Depth(backbone='resnet18'))
    if args.med_kd: models["Med KD (R18)"] = load_k(args.med_kd, Timm_Depth(backbone='resnet18'), Timm_Depth(backbone='resnet50'))
    if args.mini_base1: models["Mini1 Base (MBv3)"] = load_b(args.mini_base1, Timm_Depth(backbone='mobilenetv3_small_050'))
    if args.mini_kd1: models["Mini1 KD (MBv3)"] = load_k(args.mini_kd1, Timm_Depth(backbone='mobilenetv3_small_050'), Timm_Depth(backbone='resnet18'))
    if args.mini_base2: models["Mini2 Base (Cust)"] = load_b(args.mini_base2, Model_depth(3, 1, is_mini=True))
    if args.mini_kd2: models["Mini2 KD (Cust)"] = load_k(args.mini_kd2, Model_depth(3, 1, is_mini=True), Timm_Depth(backbone='resnet50'))

    models = {k: v for k, v in models.items() if v is not None}

    idx = random.sample(range(len(dm.val_ds)), 50)
    imgs, depths = torch.stack([dm.val_ds[i][0] for i in idx]).to(device), torch.stack([dm.val_ds[i][1] for i in idx]).to(device)

    preds_dict = {}
    with torch.no_grad():
        for name, task in models.items():
            out = task(imgs)
            preds_dict[name] = out[0] if isinstance(out, tuple) else out

    save_comparison(imgs, depths, preds_dict, n=50)
    
    mae_results = {}
    for name, preds in preds_dict.items():
        mae = torch.nn.functional.l1_loss(preds[depths>0], depths[depths>0]).item()
        mae_results[name] = mae
        print(f"MAE {name}: {mae:.4f}")

    if "T_Grande (R50)" in mae_results:
        t_mae = mae_results["T_Grande (R50)"]
        b_maes = {"Med (R18)": mae_results.get("Med Base (R18)", 0), "Mini1 (MBv3)": mae_results.get("Mini1 Base (MBv3)", 0), "Mini2 (Cust)": mae_results.get("Mini2 Base (Cust)", 0)}
        k_maes = {"Med (R18)": mae_results.get("Med KD (R18)", 0), "Mini1 (MBv3)": mae_results.get("Mini1 KD (MBv3)", 0), "Mini2 (Cust)": mae_results.get("Mini2 KD (Cust)", 0)}
        save_mae_histogram(t_mae, {k:v for k,v in b_maes.items() if v>0}, {k:v for k,v in k_maes.items() if v>0})
        print(f"\nIstogramma salvato in 'istogramma_mae.png'")

if __name__ == "__main__": main()

"""

import torch, argparse, random
import matplotlib.pyplot as plt
from Train_finale import NYUDataModule, Timm_Depth, Model_depth, BaselineTask, EncoderDistillationTask

def denormalize(t):
    return (t.cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)).clamp(0,1).permute(1,2,0).numpy()

def save_comparison(imgs, depths, preds_dict, n=10, filename="confronto_7_modelli.png"):
    n = min(n, len(imgs))
    cols = ['Input RGB'] + list(preds_dict.keys()) + ['GT']
    fig, axes = plt.subplots(n, len(cols), figsize=(3 * len(cols), n * 3))
    for i in range(n):
        axes[i, 0].imshow(denormalize(imgs[i])); axes[i, 0].axis('off')
        for j, (name, preds) in enumerate(preds_dict.items()):
            axes[i, j+1].imshow(preds[i].cpu().squeeze(), cmap='plasma'); axes[i, j+1].axis('off')
        axes[i, -1].imshow(depths[i].cpu().squeeze(), cmap='plasma'); axes[i, -1].axis('off')
        if i == 0:
            for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()

def save_error_maps(imgs, depths, preds_dict, n=10, filename="confronto_errori_pixel.png"):
    n = min(n, len(imgs))
    cols = ['Input RGB'] + list(preds_dict.keys())
    fig, axes = plt.subplots(n, len(cols), figsize=(3 * len(cols), n * 3))
    for i in range(n):
        axes[i, 0].imshow(denormalize(imgs[i])); axes[i, 0].axis('off')
        gt, mask = depths[i], depths[i] > 0
        for j, (name, preds) in enumerate(preds_dict.items()):
            err = torch.abs(preds[i] - gt)
            err[~mask] = 0 
            axes[i, j+1].imshow(err.cpu().squeeze(), cmap='inferno', vmin=0, vmax=0.5); axes[i, j+1].axis('off')
        if i == 0:
            for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t_grande", type=str)
    p.add_argument("--med_base", type=str)
    p.add_argument("--med_kd", type=str)
    p.add_argument("--mini_base1", type=str)
    p.add_argument("--mini_kd1", type=str)
    p.add_argument("--mini_base2", type=str)
    p.add_argument("--mini_kd2", type=str)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = NYUDataModule(batch_size=8, fraction=.1); dm.setup()
    
    models = {}
    def load_b(ckpt, m): return BaselineTask.load_from_checkpoint(ckpt, model=m, strict=False).to(device).eval() if ckpt else None
    def load_k(ckpt, s, t): return EncoderDistillationTask.load_from_checkpoint(ckpt, student=s, teacher=t, strict=False).to(device).eval() if ckpt else None

    if args.t_grande: models["T_Grande (R50)"] = load_b(args.t_grande, Timm_Depth(backbone='resnet50'))
    if args.med_base: models["Med Base (R18)"] = load_b(args.med_base, Timm_Depth(backbone='resnet18'))
    if args.med_kd: models["Med KD (R18)"] = load_k(args.med_kd, Timm_Depth(backbone='resnet18'), Timm_Depth(backbone='resnet50'))
    if args.mini_base1: models["Mini1 Base (MBv3)"] = load_b(args.mini_base1, Timm_Depth(backbone='mobilenetv3_small_050'))
    if args.mini_kd1: models["Mini1 KD (MBv3)"] = load_k(args.mini_kd1, Timm_Depth(backbone='mobilenetv3_small_050'), Timm_Depth(backbone='resnet18'))
    if args.mini_base2: models["Mini2 Base (Cust)"] = load_b(args.mini_base2, Model_depth(3, 1, is_mini=True))
    if args.mini_kd2: models["Mini2 KD (Cust)"] = load_k(args.mini_kd2, Model_depth(3, 1, is_mini=True), Timm_Depth(backbone='resnet50'))

    models = {k: v for k, v in models.items() if v is not None}

    idx = random.sample(range(len(dm.val_ds)), 50)
    imgs, depths = torch.stack([dm.val_ds[i][0] for i in idx]).to(device), torch.stack([dm.val_ds[i][1] for i in idx]).to(device)

    preds_dict = {}
    with torch.no_grad():
        for name, task in models.items():
            out = task(imgs)
            preds_dict[name] = out[0] if isinstance(out, tuple) else out

    save_comparison(imgs, depths, preds_dict, n=50)
    save_error_maps(imgs, depths, preds_dict, n=50)
    
    for name, preds in preds_dict.items():
        print(f"MAE {name}: {torch.nn.functional.l1_loss(preds[depths>0], depths[depths>0]).item():.4f}")

if __name__ == "__main__": main()