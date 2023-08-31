import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math
import SimpleITK as sitk
import numpy as np
from blind_deconv import plot_3d_image
from pytorch_msssim import ssim, ms_ssim


mse = torch.nn.MSELoss()
mae = nn.L1Loss()

class Kernel_Loss(nn.Module):
    def __init__(self):
        super(Kernel_Loss, self).__init__()
    def forward(self, k,y):
        #l1_loss = nn.L1Loss()(k,y)
        mse_loss = nn.MSELoss()(k,y) * 100000
        return mse_loss    

class Deconv_Loss(nn.Module):
    def __init__(self, tv_weight, ct_weight, ct_tex, device):
        super(Deconv_Loss, self).__init__()
        self.tv_weight = tv_weight
        self.ct_tex=ct_tex
        self.ct_weight = ct_weight
        self.kernel_weight = 10e-1
        self.dose_weight = 10e-5
        self.curvature_weight = 0.01
        self.centre_weight = 0.01
    def forward(self, prediction, target, kernel, deblurred_prediction, step):
        if step < 2000:

            # dose_penalty = torch.abs(torch.sum(target)-torch.sum(prediction))

            fidelity_term = mae(prediction, target)+ mse(prediction, target)

            loss_kernel = torch.norm(kernel[kernel > 0.7], p=2)  


            loss = fidelity_term + (self.kernel_weight * loss_kernel) 

            if step < 300:
                _, _, k_z, k_y, k_x = kernel.shape
                grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(k_z), torch.arange(k_y), torch.arange(k_x))
                grid_z = grid_z.to(kernel.device)
                grid_y = grid_y.to(kernel.device)
                grid_x = grid_x.to(kernel.device)
                distance_z = torch.abs(grid_z -int(k_z/2))
                distance_y = torch.abs(grid_y - int(k_y/2))
                distance_x = torch.abs(grid_x - int(k_x/2))
                centre_loss = kernel * (distance_z + distance_y + distance_x)
                centre_loss = centre_loss.sum()
                

                jep = get_joint_entropy_penalty(deblurred_prediction, self.ct_tex)    #penalty for both the high and low uptake regions (better probability distributions)
                jep = jep
                loss = loss + (centre_loss * self.centre_weight ) + jep * self.ct_weight #+ dose_penalty * self.dose_weight

            if step % 25 == 0:
                if (self.kernel_weight * loss_kernel) > 0.4*fidelity_term.item():
                    self.kernel_weight = 0.01 * fidelity_term.item() / loss_kernel.item()   #make weight 1% of fidelity term
                if step < 300 and (centre_loss * self.centre_weight) > 0.1*fidelity_term.item():
                    self.centre_weight = 0.01 * fidelity_term.item() / centre_loss.item()   #make weight 1% of fidelity term    
                if step < 300 and (self.ct_weight * jep) > 0.01*fidelity_term.item():
                    self.ct_weight = 0.001 * fidelity_term.item() / jep.item()   #make weight 1% of fidelity term
                # if (self.dose_weight * dose_penalty) > 0.005*fidelity_term.item():
                #     self.dose_weight = 0.0001 * fidelity_term.item() / dose_penalty.item()   #make weight 1% of fidelity term    

                # if (self.curvature_weight * loss_curvature) > 0.02*fidelity_term.item():
                #     self.curvature_weight =  0.005 * fidelity_term.item() / loss_curvature.item()   #make weight 1% of fidelity term

                
            if step % 100 == 0:
                print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)}")# || curvature loss: {round(self.curvature_weight*loss_curvature.item(),8)}")# || dose loss: {round(self.dose_weight*dose_penalty.item(),8)} ")


        elif step < 3500:   #introduce tv loss and make kernel loss weaker
            fidelity_term = mae(prediction, target) + mse(prediction, target)#1-ms_ssim(pad_if_small(prediction), pad_if_small(target), data_range=1)#mse(prediction, target) + mae(prediction, target)
            # dose_penalty = torch.abs(torch.sum(target)-torch.sum(prediction))

            # # Compute the TV regularization term
            grad_x = torch.abs(deblurred_prediction[:,:,:,:,:-1] - deblurred_prediction[:,:,:,:,1:])
            grad_y = torch.abs(deblurred_prediction[:,:,:,:-1,:] - deblurred_prediction[:,:,:,1:,:])
            grad_z = torch.abs(deblurred_prediction[:,:,:-1,:,:] - deblurred_prediction[:,:,1:,:,:])
            tv_loss = (torch.mean(grad_x) + torch.mean(grad_y) + torch.mean(grad_z))
            #dose_penalty = torch.abs(torch.sum(target)-torch.sum(prediction))
            #curvature penal
            
            # gradients = torch.gradient(deblurred_prediction, dim=(2,3,4))
            # curvature = torch.zeros_like(deblurred_prediction)
            # for grad in gradients:
            #     curve = torch.gradient(grad, dim=(2,3,4))
            #     for c in curve:
            #         curvature += torch.abs(c)
            # loss_curvature = torch.mean(curvature)
            
            #small_penalty = torch.sum(kernel[kernel < 0.01])
            norm_penalty =torch.norm(kernel[kernel > 0.5], p=2)
            #tv_penalty = tv_kernel * self.tv_weight
            loss_kernel =  norm_penalty 
   
            loss = fidelity_term + (self.kernel_weight * loss_kernel) + (self.tv_weight * tv_loss)#+ self.dose_weight*dose_penalty #+ (self.tv_weight * tv_loss) #+ (self.dose_weight * dose_penalty)   #+ (self.curvature_weight * loss_curvature) + (self.tv_weight * tv_loss
            
            if step % 25 == 0:
                if (self.kernel_weight * loss_kernel) > 0.05*fidelity_term.item():
                    self.kernel_weight = 0.001 * fidelity_term.item() / loss_kernel.item()   #make weight 1% of fidelity term

                if (self.tv_weight * tv_loss) > 0.01*fidelity_term.item():
                    self.tv_weight = 0.001 * fidelity_term.item() / tv_loss.item()   #make weight 1% of fidelity term

                # if (self.curvature_weight * loss_curvature) > 0.01*fidelity_term.item():
                #     self.curvature_weight = 0.001 * fidelity_term.item() / loss_curvature.item()   #make weight 1% of fidelity term

                # if (self.dose_weight * dose_penalty) > 0.005*fidelity_term.item():
                #     self.dose_weight = 0.0001 * fidelity_term.item() / dose_penalty.item()   #make weight 1% of fidelity term
            if step % 100 == 0:
                print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)}")# || tv loss: {round(self.tv_weight*tv_loss.item(),8)}")
            #print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)} || curvature loss: {round(self.curvature_weight*loss_curvature.item(),8)} || dose loss: {round(self.dose_weight*dose_penalty.item(),8)} || tv loss: {round(self.tv_weight*tv_loss.item(),8)}")


        else:   #introduce tv loss and make kernel loss weaker
            fidelity_term = 1-ms_ssim(pad_if_small(prediction), pad_if_small(target), data_range=1)#mse(prediction, target) + mae(prediction, target)
            
            #dose_penalty = torch.abs(torch.sum(target)-torch.sum(prediction))

            # Compute the TV regularization term
            grad_x = torch.abs(deblurred_prediction[:,:,:,:,:-1] - deblurred_prediction[:,:,:,:,1:])
            grad_y = torch.abs(deblurred_prediction[:,:,:,:-1,:] - deblurred_prediction[:,:,:,1:,:])
            grad_z = torch.abs(deblurred_prediction[:,:,:-1,:,:] - deblurred_prediction[:,:,1:,:,:])
            tv_loss = (torch.mean(grad_x) + torch.mean(grad_y) + torch.mean(grad_z))

            loss = fidelity_term + (self.tv_weight * tv_loss) #
            if step % 25 == 0:


                if (self.tv_weight * tv_loss) > 0.01*fidelity_term.item():
                    self.tv_weight = 0.001 * fidelity_term.item() / tv_loss.item()   #make weight 1% of fidelity term

        return loss#, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
    

#just for the pre-training multiscale loops
class Deconv_Loss_Small(nn.Module):
    def __init__(self, tv_weight, ct_weight, ct_tex, device):
        super(Deconv_Loss_Small, self).__init__()
        self.tv_weight = tv_weight
        self.ct_tex=ct_tex
        self.ct_weight = ct_weight
        self.kernel_weight = 10e-1
        self.dose_weight = 10e-5
        self.centre_weight = 0.01
        self.curvature_weight = 0.01
    def forward(self, prediction, target, kernel, deblurred_prediction, step):
  

        fidelity_term = mse(prediction, target)+ mae(prediction, target)

        loss_kernel =  torch.norm(kernel[kernel > 0.7], p=2) #+ 0.001*torch.norm(kernel[kernel < 0.001], p=2)
        
        loss = fidelity_term + (self.kernel_weight * loss_kernel) #+ (self.curvature_weight * loss_curvature) #+ (self.dose_weight * dose_penalty)  


        if step < 300: 

            _, _, k_z, k_y, k_x = kernel.shape
            grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(k_z), torch.arange(k_y), torch.arange(k_x))
            grid_z = grid_z.to(kernel.device)
            grid_y = grid_y.to(kernel.device)
            grid_x = grid_x.to(kernel.device)
            distance_z = torch.abs(grid_z -int(k_z/2))
            distance_y = torch.abs(grid_y - int(k_y/2))
            distance_x = torch.abs(grid_x - int(k_x/2))
            centre_loss = kernel * (distance_z + distance_y + distance_x)
            centre_loss = centre_loss.sum()

            loss = loss + (centre_loss * self.centre_weight )
            loss = loss + self.kernel_weight*0.001*torch.norm(kernel[kernel < 0.001], p=2)  
  

        if step % 25 == 0:
            if (self.kernel_weight * loss_kernel) > 0.4*fidelity_term.item():
                self.kernel_weight = 0.01 * fidelity_term.item() / loss_kernel.item()   #make weight 1% of fidelity term
            
            if step < 300 and (centre_loss * self.centre_weight) > 0.1*fidelity_term.item():
                self.centre_weight = 0.01 * fidelity_term.item() / centre_loss.item()   #make weight 1% of fidelity term
            


        if step % 100 == 0:
            print(f"fidelity loss: {round(fidelity_term.item(),8)}|| kernel loss: {round(self.kernel_weight*loss_kernel.item(),8)}")
        return loss#, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel    
  

def get_joint_entropy_penalty(x,y):
    #here implement the joint entropy loss term of the pet prediction and the ct texture map (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6853071/)
    #will include the high region, where psma uptake concentrates, as well as the low region which is the rest of the body where levels are low. 
    # This makes the probability distributions less concentrated in a single region
    if x.shape[-1] < 161:
        x = pad_if_small(x)
        y = pad_if_small(y)
    
    x = x.double()
    y = y.double()


    je = 1 - ms_ssim(x, y, data_range=1)
    x = x.float()
    y = y.float()

    return je

def pad_if_small(image):
    _,_,z_,n,_ = image.size()
    if n > 160:
        return image
    padding_n = (162-n+1) // 2

    padded_img = F.pad(image, (padding_n, padding_n,padding_n, padding_n)) 
    return padded_img