import torch, argparse, time
from src.models.models import Timm_Depth, Model_depth

def evaluate_efficiency(model, device, img_size=(1, 3, 256, 256)):
    model.to(device).eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    dummy = torch.randn(img_size).to(device)
    
    with torch.no_grad():
        for _ in range(10): _ = model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(50): _ = model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    avg_ms = ((time.perf_counter() - start) / 50) * 1000
    return params, avg_ms, 1000 / avg_ms if avg_ms > 0 else 0

def main():
    p = argparse.ArgumentParser(description="Benchmark Hardware (Solo Architetture)")
    p.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    if args.device == "cpu" and args.threads > 0: torch.set_num_threads(args.threads)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Inizializziamo solo le architetture vuote (i pesi non cambiano i tempi di calcolo)
    unique_architectures = {
        "Teacher (ResNet50)": Timm_Depth(backbone='resnet50'),
        "Medium (ResNet18)": Timm_Depth(backbone='resnet18'),
        "Medium (ResNet18)": Model_depth(3, 1, is_mini=False),
        "Mini 1 (MBv3_100)": Timm_Depth(backbone='mobilenetv3_small_100'),
        "Mini 2 (Custom)": Model_depth(3, 1, is_mini=True)
    }

    print(f"\n🚀 SIMULAZIONE EDGE | Device: {device.type.upper()} | Threads: {args.threads if device.type == 'cpu' else 'N/A'}")
    print(f"\n{'Architettura':<20} | {'Parametri (M)':<15} | {'Inferenza (ms)':<15} | {'FPS':<10}")
    print("-" * 68)
    
    for name, model in unique_architectures.items():
        p_m, t_ms, fps = evaluate_efficiency(model, device)
        print(f"{name:<20} | {p_m:<15.2f} | {t_ms:<15.2f} | {fps:<10.1f}")

if __name__ == "__main__": main()