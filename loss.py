import torch
import torch.nn as nn
import torch.nn.functional as F

def l1_loss(x1, x2, mask):
#     size = mask.size()
#     masksum = mask.view(size[0], size[1], -1).sum(-1, keepdim=True) + 1
#     diffs = torch.abs(mask*(x1-x2)).view(size[0], size[1], -1).sum(-1, keepdim=True)
#     diffs = torch.sum(diffs/masksum, 1)
#     return torch.mean(diffs)
    return F.l1_loss(x1[mask], x2[mask])

def compute_img_stats(img):
    # the padding is to maintain the original size
    img_pad = F.pad(img, (1,1,1,1), mode='reflect')
    mu = F.avg_pool2d(img_pad, kernel_size=3, stride=1, padding=0)
    sigma = F.avg_pool2d(img_pad**2, kernel_size=3, stride=1, padding=0) - mu**2
    return mu, sigma

def compute_SSIM(img0, img1):
    mu0, sigma0= compute_img_stats(img0) 
    mu1, sigma1= compute_img_stats(img1)
    # the padding is to maintain the original size
    img0_img1_pad = F.pad(img0 * img1, (1,1,1,1), mode='reflect')
    sigma01 = F.avg_pool2d(img0_img1_pad, kernel_size=3, stride=1, padding=0) - mu0*mu1

    C1 = .0001
    C2 = .0009

    ssim_n = (2*mu0*mu1 + C1) * (2*sigma01 + C2)
    ssim_d = (mu0**2 + mu1**2 + C1) * (sigma0 + sigma1 + C2)
    ssim = ssim_n / ssim_d
    return ((1-ssim)*.5).clamp(0, 1)

def ssim_loss(img0, img1, mask):
    return torch.mean(compute_SSIM(img0, img1)[mask])

class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def grad_x(self, target):
        return target[:,:,:,:-1] - target[:,:,:,1:]
    
    def grad_y(self, target):
        return target[:,:,:-1] - target[:,:,1:]
    
    def forward(self, imgs, ds, mask):
        img_grad_y = self.grad_y(imgs)
        img_grad_x = self.grad_x(imgs)
        
#         normalize the depth map to make the magnitude be consistance
        ds = ds / torch.mean( torch.mean(ds, dim=2, keepdim=True), dim=3, keepdim=True)
        
        disp_grad_y = self.grad_y(ds)
        disp_grad_x = self.grad_x(ds)
        
        weight_x = torch.mean( torch.exp( -torch.abs(img_grad_x)), dim=1, keepdim=True ) 
        weight_y = torch.mean( torch.exp( -torch.abs(img_grad_y)), dim=1, keepdim=True ) 
        
        loss_x = (disp_grad_x).pow(2) * weight_x
        loss_y = (disp_grad_y).pow(2) * weight_y

        mask_x = (mask[:,0,:,:-1]+mask[:,0,:,1:])>0.0
        mask_y = (mask[:,0,:-1]+mask[:,0,1:])>0.0
        
#         pdb.set_trace()
        return torch.mean(loss_x[mask_x.unsqueeze(1)]) + torch.mean(loss_y[mask_y.unsqueeze(1)])
