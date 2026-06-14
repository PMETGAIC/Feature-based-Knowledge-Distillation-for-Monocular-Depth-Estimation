import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        
    def forward(self, pred, target, mask):
        if not mask.any(): return torch.tensor(0.0, device=pred.device, requires_grad=True)
        l1 = F.l1_loss(pred[mask], target[mask])
        p, t = pred*mask, target*mask
        mu_p, mu_t = F.avg_pool2d(p, 3, 1, 1), F.avg_pool2d(t, 3, 1, 1)            
        sig_p = F.avg_pool2d(p**2, 3, 1, 1) - mu_p**2
        sig_t = F.avg_pool2d(t**2, 3, 1, 1) - mu_t**2
        sig_pt = F.avg_pool2d(p*t, 3, 1, 1) - mu_p*mu_t
        ssim = ((2*mu_p*mu_t + 1e-4)*(2*sig_pt + 1e-4)) / ((mu_p**2 + mu_t**2 + 1e-4)*(sig_p + sig_t + 1e-4))
        return self.alpha*l1 + self.beta*torch.clamp((1-ssim)/2, 0, 1)[mask].mean()

def relational_loss(s, t):
    s_pool, t_pool = F.normalize(F.adaptive_avg_pool2d(s, (16,16)).flatten(2), dim=1), F.normalize(F.adaptive_avg_pool2d(t, (16,16)).flatten(2), dim=1)
    return F.mse_loss(s_pool.transpose(1,2) @ s_pool, t_pool.transpose(1,2) @ t_pool)

def at_loss(s, t):
    s_map, t_map = F.interpolate(s.pow(2).mean(1, keepdim=True), size=t.shape[2:], mode='bilinear'), t.pow(2).mean(1, keepdim=True)
    return F.mse_loss(F.normalize(s_map.flatten(1), dim=1), F.normalize(t_map.flatten(1), dim=1))

