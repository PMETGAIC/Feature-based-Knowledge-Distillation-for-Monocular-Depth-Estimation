import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from Train_finale import *

def objective(trial):
    w_at = trial.suggest_float("w_at", 0.1, 5.0)
    w_rel = trial.suggest_float("w_rel", 0.1, 5.0)
    w_out = trial.suggest_float("w_out", 0.1, 5.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    student = Timm_Depth(backbone='mobilenetv3_small_050') 
    teacher = Timm_Depth(backbone='resnet18') 
    
    model = EncoderDistillationTask(
        student=student, 
        teacher=teacher, 
        lr=lr, 
        w_at=w_at, 
        w_rel=w_rel, 
        w_out=w_out
    )

    dm = NYUDataModule(batch_size=32, fraction=0.1)
    
    trainer = pl.Trainer(
        max_epochs=5, 
        accelerator="auto",
        enable_checkpointing=False,
        logger=False,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/mae")]
    )

    trainer.fit(model, datamodule=dm)
    return trainer.callback_metrics["val/mae"].item()

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50, timeout=3600)
    
    print(study.best_value)
    print(study.best_params)