import torch
import torch.nn.functional as F

def img_MSE(recon, raw):
    with torch.no_grad(): 
        raw, recon = raw.type(torch.float64), recon.type(torch.float64)
        return F.mse_loss(raw,recon,reduction='mean').item()

def img_PSNR(recon, raw):  # otp: batch of 2-D features  target: batch of 1-D raw audios
    with torch.no_grad(): 
        raw, recon = raw.type(torch.float64), recon.type(torch.float64)
        mse = F.mse_loss(raw,recon,reduction='mean')
        max_I = torch.max(torch.amax(raw),torch.amax(recon))
        return 20*(torch.log10(max_I / (torch.sqrt(mse)))).item()