import os, torch, argparse, random, numpy as np
import matplotlib.pyplot as plt
from src.training.train import NYUDataModule, Timm_Depth, Model_depth, BaselineTask, EncoderDistillationTask

def denormalize(t):
    return (t.cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)).clamp(0,1).permute(1,2,0).numpy()

def save_comparison(imgs, depths, preds_dict, n=10, filename="figures/comparison.png"):
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

def save_error_maps(imgs, depths, preds_dict, n=10, filename="figures/pixels_error.png"):
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

def save_mae_histogram(t_mae, b_maes, k_maes, filename="figures/istogramma_mae.png"):
    labels = list(b_maes.keys())
    if not labels: return
    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    
    b_pct = [(b_maes[k] / t_mae) * 100 if b_maes[k] > 0 else 0 for k in labels]
    k_pct = [(k_maes[k] / t_mae) * 100 if k_maes[k] > 0 else 0 for k in labels]
    
    r1 = ax.bar(x - w/2, b_pct, w, label='Baseline', color='#3498db', edgecolor='none')
    r2 = ax.bar(x + w/2, k_pct, w, label='Student (KD)', color='#e74c3c', edgecolor='none')
    
    ax.axhline(100, color='#2c3e50', linestyle='--', linewidth=1.2, label=f'Teacher ({t_mae:.4f} = 100%)', zorder=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(left=False, bottom=False)
    
    ax.set_ylabel('% Errore Relativo al Teacher', fontsize=11, color='#555555')
    ax.set_title('Impatto del KD: Errore rispetto al Teacher', fontsize=14, color='#333333', pad=15)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, color='#333333')
    ax.legend(frameon=False, loc='upper left'); ax.set_axisbelow(True)
    plt.grid(axis='y', linestyle='-', alpha=0.15, color='gray')
    
    for r, p, v in zip(r1, b_pct, b_maes.values()):
        if v > 0: ax.text(r.get_x() + w/2, p - 4, f'{p:.1f}%\n({v:.4f})', ha='center', va='top', fontsize=9.5, fontweight='bold', color='white')
    for r, p, v in zip(r2, k_pct, k_maes.values()):
        if v > 0: ax.text(r.get_x() + w/2, p - 4, f'{p:.1f}%\n({v:.4f})', ha='center', va='top', fontsize=9.5, fontweight='bold', color='white')
    
    ax.set_ylim(0, max(max(b_pct), max(k_pct)) * 1.15)
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

    os.makedirs("figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = NYUDataModule(batch_size=8, fraction=.1); dm.setup()
    
    models = {}
    def load_b(ckpt, m): return BaselineTask.load_from_checkpoint(ckpt, model=m, strict=False).to(device).eval() if ckpt else None
    def load_k(ckpt, s, t): return EncoderDistillationTask.load_from_checkpoint(ckpt, student=s, teacher=t, strict=False).to(device).eval() if ckpt else None

    if args.t_grande: models["T_Grande (R50)"] = load_b(args.t_grande, Timm_Depth(backbone='resnet50'))
    if args.med_base: models["Med Base (R18)"] = load_b(args.med_base, Timm_Depth(backbone='resnet18'))
    if args.med_kd: models["Med KD (R18)"] = load_k(args.med_kd, Timm_Depth(backbone='resnet18'), Timm_Depth(backbone='resnet50'))
    if args.mini_base1: models["Mini1 Base (MBv3)"] = load_b(args.mini_base1, Timm_Depth(backbone='mobilenetv3_small_100'))
    if args.mini_kd1: models["Mini1 KD (MBv3)"] = load_k(args.mini_kd1, Timm_Depth(backbone='mobilenetv3_small_100'), Timm_Depth(backbone='resnet18'))
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
    
    mae_results = {}
    for name, preds in preds_dict.items():
        mae = torch.nn.functional.l1_loss(preds[depths>0], depths[depths>0]).item()
        mae_results[name] = mae
        print(f"MAE {name}: {mae:.4f}")

    if "T_Grande (R50)" in mae_results:
        t_mae = mae_results["T_Grande (R50)"]
        pairs = [("Med (R18)", "Med Base (R18)", "Med KD (R18)"), ("Mini1 (MBv3)", "Mini1 Base (MBv3)", "Mini1 KD (MBv3)"), ("Mini2 (Cust)", "Mini2 Base (Cust)", "Mini2 KD (Cust)")]
        b_maes, k_maes = {}, {}
        for label, b_key, k_key in pairs:
            if b_key in mae_results or k_key in mae_results:
                b_maes[label] = mae_results.get(b_key, 0.0)
                k_maes[label] = mae_results.get(k_key, 0.0)
        
        if b_maes: 
            save_mae_histogram(t_mae, b_maes, k_maes)
            print("Grafici aggiuntivi salvati nella cartella 'figures'.")

if __name__ == "__main__": main()