import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from src.training.losses import PaperLoss, relational_loss, at_loss

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
    def __init__(self, student, teacher, lr=1e-4, w_at=1.066, w_rel=3.261, w_out=1.443):
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