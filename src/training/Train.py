import os, argparse, torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping 
from src.datasets.dataset import NYUDataModule
from src.models.models import Model_depth, Timm_Depth
from src.training.tasks import BaselineTask, EncoderDistillationTask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baseline", "kd"], default="baseline")
    parser.add_argument("--mode", choices=["train", "resume", "finetune"], default="train")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--t_ckpt", type=str)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--use_custom_model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = f"./experiments/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    if args.teacher:
        fraction = 1.0
    elif args.mini:
        fraction = 0.1
    else:
        fraction = 0.5

    dm = NYUDataModule(batch_size=32, fraction=fraction)
    dm.setup()

    if args.task == "baseline":
        model = Model_depth(3, 1, is_mini=args.mini) if args.use_custom_model else Timm_Depth(backbone='resnet50' if args.teacher else ('mobilenetv3_small_100' if args.mini else 'resnet18'))
        task = BaselineTask.load_from_checkpoint(args.ckpt, model=model) if args.mode == "finetune" and args.ckpt else BaselineTask(model)
    else:
        student_base = Model_depth(3, 1, is_mini=args.mini) if args.use_custom_model else Timm_Depth(backbone='mobilenetv3_small_100' if args.mini else 'resnet18')
        t_model = Timm_Depth(backbone='resnet18' if (args.mini or (args.t_ckpt and 'TA' in args.t_ckpt)) else 'resnet50')
        
        if args.t_ckpt:
            ckpt = torch.load(args.t_ckpt, map_location="cpu")
            t_sd = {k.replace("student.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("student.")} if any(k.startswith("student.") for k in ckpt["state_dict"].keys()) else {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            t_model.load_state_dict(t_sd, strict=False)
            
        task = EncoderDistillationTask.load_from_checkpoint(args.ckpt, student=student_base, teacher=t_model, strict=False, lr=1e-4) if args.mode == "finetune" and args.ckpt else EncoderDistillationTask(student_base, t_model, lr=1e-4)

    csv_logger = CSVLogger(save_dir=save_dir, name=args.task)
    ckpt_cb = ModelCheckpoint(monitor="val/mae", mode="min", save_top_k=1, filename="best-{epoch:02d}-{val/mae:.4f}")
    es_cb = EarlyStopping(monitor="val/mae", min_delta=0.00, patience=10, verbose=False, mode="min")
    
    trainer = pl.Trainer(max_epochs=args.epoch, accelerator="auto", precision="16-mixed", accumulate_grad_batches=1, gradient_clip_val=1.0, default_root_dir=save_dir, callbacks=[ckpt_cb, es_cb], logger=csv_logger)
    
    trainer.fit(task, dm, ckpt_path=args.ckpt if args.mode == "resume" else None)
    trainer.validate(task, dm)

if __name__ == "__main__":
    main()