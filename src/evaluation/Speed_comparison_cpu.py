import torch
import time
from src.training.train import Timm_Depth, Model_depth, BaselineTask, EncoderDistillationTask

def benchmark_cpu(model, name, img_size=(1, 3, 256, 256), iterations=100):
    # Simula l'ambiente Edge: solo CPU e max 4 thread (come un Raspberry Pi 4)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(img_size).to(device)
    
    with torch.no_grad(): # warm up
        for _ in range(10):
            _ = model(dummy_input)
            
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end_time = time.perf_counter()
    
    # Calcolo metriche
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    fps = 1000 / avg_latency_ms
    
    print(f"--- Risultati Simlazione Edge: {name} ---")
    print(f"Latenza media: {avg_latency_ms:.2f} ms")
    print(f"Frame Per Second (FPS): {fps:.2f}\n")

if __name__ == "__main__":
    print("Inizio Edge Simulation Benchmark (CPU - 4 Threads)...\n")
    
    t_grande = Timm_Depth(backbone='resnet50')
    med_base = Timm_Depth(backbone='resnet18')
    mini_base = Timm_Depth(backbone='mobilenetv3_small_100') 
    
    benchmark_cpu(t_grande, "Teacher (ResNet-50)")
    benchmark_cpu(med_base, "Student Base (ResNet-18)")
    benchmark_cpu(mini_base, "Mini Student (MobileNetV3)")