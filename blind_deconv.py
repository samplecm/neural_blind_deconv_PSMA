#this code is used for the work: "Neural blind deconvolution for simultaneous partial volume effect correction and super-sampling of PSMA PET images" 2023
#By Caleb Sample, Carlose Uribe, Arman Rahmim, François Benard, Jonn Wu, Hal Clark


import os
import numpy as np
import torch
import torch.optim
import SimpleITK as sitk
import pickle
import glob
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import model_deconv
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_fill_holes, label, zoom 
from utils import *
import loss_functions
import model_deconv
import torch.nn.functional as F
import radiomics
import time
import six 
import cv2
from copy import deepcopy

torch.cuda.empty_cache()

def main():
    data_folder = os.path.join(os.getcwd(), "data_deblur")
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient_num in patient_nums[0:]:
        print(f"Loading data for for {patient_num}...")

        img_series_path = os.path.join(data_folder, patient_num, "img_dict")


        #load your image and mask
        with open(img_series_path, "rb") as fp:
            img_dict = pickle.load(fp)  #see image_processing.py Img_Series func for how this was formatted
        with open(os.path.join(data_folder, patient_num, "mask_dict"), "rb") as fp:
            mask_dict = pickle.load(fp)["PET"]   #this is a dictionary where keys are structure names, which themselves are contours objects which have processed mask arrays. 



        #get min / max parotid voxels
        z_min = 1000
        z_max = -1000
        x_min = 1000
        x_max = -1000
        y_min = 1000
        y_max = -1000
        for structure in mask_dict:
            mask = mask_dict[structure].whole_roi_masks
            z_min_mask, z_max_mask = np.min(np.where(mask)[0]),  np.max(np.where(mask)[0]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if z_min_mask < z_min:
                z_min = z_min_mask
            if z_max_mask > z_max:
                z_max = z_max_mask  

            y_min_mask, y_max_mask = np.min(np.where(mask)[1]),  np.max(np.where(mask)[1]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if y_min_mask < y_min:
                y_min = y_min_mask
            if y_max_mask > y_max:
                y_max = y_max_mask 

            x_min_mask, x_max_mask = np.min(np.where(mask)[2]),  np.max(np.where(mask)[2]) #[int(pet_img.shape[0]*0.3), int(pet_img.shape[0]*0.7)]
            if x_min_mask < x_min:
                x_min = x_min_mask
            if x_max_mask > x_max:
                x_max = x_max_mask 
    
        print("Resampling CT image and computing texture feature map...")   
        
        #get the images into aligned arrays
        pet_img, ct_img, ct_tex = prepare_pet_ct_images(img_dict, patient_num)

        xy_min = min(x_min, y_min)-10    #extend window around parotids
        xy_max = max(x_max,y_max)+10
        if (xy_max - xy_min) % 2 == 1:  #need even dimensions
            xy_min -= 1
        if (z_max - z_min ) % 2 == 1:
            z_min -= 1    

        #crop images
        pet_img_region = pet_img[z_min-6:z_max+6,xy_min-10:xy_max-10,xy_min:xy_max]
        ct_tex = ct_tex[z_min-6:z_max+6,xy_min-10:xy_max-10,xy_min:xy_max]


        #coords arrays contain the cartesian coordinates of each image voxel, used for mask calculations
        img_dict["PET"].deconv_coords = img_dict["PET"].coords_array[:,z_min-6:z_max+6,xy_min-10:xy_max-10,xy_min:xy_max]
        #upsample the coords array to match deblurred image
        new_coords = np.zeros((img_dict["PET"].deconv_coords.shape[0], img_dict["PET"].deconv_coords.shape[1]*2, img_dict["PET"].deconv_coords.shape[2]*2, img_dict["PET"].deconv_coords.shape[3]*2))
        new_coords[0,:,:,:] = zoom(img_dict["PET"].deconv_coords[0,:,:,:], zoom=2, order=1)
        new_coords[1,:,:,:] = zoom(img_dict["PET"].deconv_coords[1,:,:,:], zoom=2, order=1)
        new_coords[2,:,:,:] = zoom(img_dict["PET"].deconv_coords[2,:,:,:], zoom=2, order=1)
        img_dict["PET"].deconv_coords_img = img_dict["PET"].coords_array_img[:,z_min-6:z_max+6,xy_min-10:xy_max-10,xy_min:xy_max]
        #upsample the coords array to match deblurred image
        new_coords_img = np.zeros((img_dict["PET"].deconv_coords_img.shape[0], img_dict["PET"].deconv_coords_img.shape[1]*2, img_dict["PET"].deconv_coords_img.shape[2]*2, img_dict["PET"].deconv_coords_img.shape[3]*2))
        new_coords_img[0,:,:,:] = zoom(img_dict["PET"].deconv_coords_img[0,:,:,:], zoom=2, order=1)
        new_coords_img[1,:,:,:] = zoom(img_dict["PET"].deconv_coords_img[1,:,:,:], zoom=2, order=1)
        new_coords_img[2,:,:,:] = zoom(img_dict["PET"].deconv_coords_img[2,:,:,:], zoom=2, order=1)
        img_dict["PET"].deconv_coords_2x = new_coords
        img_dict["PET"].deconv_coords_2x_img = new_coords_img

        #run method
        prediction_blurred, prediction_deblurred, kernel = blind_deconvolution(pet_img_region, ct_tex, patient_num, load_models=False)


        #save results
        img_dict["PET"].deconv_array = prediction_deblurred
        img_dict["PET"].deconv_array_blurred = prediction_blurred
        img_dict["PET"].kernel = kernel
        img_dict["PET"].ct_texture_map = ct_tex

        
        #img_dict["PET"].deconv_coords_img = img_dict["CT"].coords_array_img[z_min-2:z_max+6,xy_min-10:xy_max-10,xy_min:xy_max]

        with open(img_series_path, "wb") as fp:
            pickle.dump(img_dict, fp)
    return


def prepare_pet_ct_images(img_dict, patient_num, try_load=True):
    if try_load:
        try:
            with open(os.path.join(os.getcwd(), "cache", f"{patient_num}_training_images"), "rb") as fp:
                print("Loaded prepared training images from cache.")
                return pickle.load(fp)
        except:
            print("Failed to load images from cache. Calculating from loaded image dictionaries..")    
    
    img_series_pet = img_dict["PET"]
    suv_factors = img_series_pet.suv_factors
    img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     
    img_ct = img_dict["CT"].image_array
    coords_ct = img_dict["CT"].coords_array
    coords_pet = img_series_pet.coords_array
    # plot_3d_image(img_pet)
    # plot_3d_image(img_ct)
    #get the physical space of each image for resampling pet
    origin_pet = (float(coords_pet[0,0,0,0]), float(coords_pet[1,0,0,0]), float(coords_pet[2,0,0,0]))
    origin_ct = (float(coords_ct[0,0,0,0]), float(coords_ct[1,0,0,0]), float(coords_ct[2,0,0,0]))
    
    spacing_pet = np.array([img_dict["PET"].pixel_spacing[0], img_dict["PET"].pixel_spacing[1],img_dict["PET"].slice_thickness])
    spacing_ct = np.array([img_dict["CT"].pixel_spacing[0], img_dict["CT"].pixel_spacing[1], img_dict["CT"].slice_thickness])

    direction = (1,0,0,0,1,0,0,0,1)

    img_ct_sitk = sitk.GetImageFromArray(img_ct)
    img_pet_sitk = sitk.GetImageFromArray(img_pet)

    img_ct_sitk.SetSpacing(spacing_ct)
    img_pet_sitk.SetSpacing(spacing_pet)
    img_ct_sitk.SetDirection(direction)
    img_pet_sitk.SetDirection(direction)
    img_ct_sitk.SetOrigin(origin_ct)
    img_pet_sitk.SetOrigin(origin_pet)
    
    img_ct_sitk = sitk.Resample(img_ct_sitk,img_pet_sitk, sitk.Transform(), sitk.sitkLinear, 0, img_ct_sitk.GetPixelID())    #double the size of the pet image to match the ct
    pet_img = sitk.GetArrayFromImage(img_pet_sitk)   #convert to numpy arrays
    ct_img = sitk.GetArrayFromImage(img_ct_sitk)

    #also want to get the glrlm of the ct image.
    mask = np.ones_like(ct_img)
    mask[pet_img < 1e-2] = 0
    labels, _ = label(mask)
    sizes = np.bincount(labels.flatten())
    largest_label = np.argmax(sizes[1:]) + 1 #ignore background term
    mask = labels == largest_label
    mask = binary_dilation(binary_erosion(mask, structure=np.ones((6,6,6))),structure=np.ones((6,6,6)))
    mask = binary_erosion(binary_dilation(mask, structure=np.ones((8,8,8))),structure=np.ones((8,8,8)))
    mask[ct_img == 0] = 0
    mask = binary_fill_holes(mask)
    #plot_3d_image(pet_img)
    #plot_3d_image(mask)
    mask = sitk.GetImageFromArray(mask.astype(int))
    mask.SetSpacing(spacing_ct)
    mask.SetDirection(direction)
    mask.SetOrigin(origin_ct)
    mask = sitk.Resample(mask,img_pet_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())    #double the size of the pet image to match the ct

    ct_tex = get_glrlm_voxel_map(img_ct_sitk, mask=mask, name=img_dict["PET"].name, try_load=True)
    
    with open(os.path.join(os.getcwd(), "cache", f"{patient_num}_training_images"), "wb") as fp:
        pickle.dump([pet_img,ct_img, ct_tex], fp)
    return pet_img, ct_img, ct_tex

def get_glrlm_voxel_map(img: sitk.Image, mask: sitk.Image, name, try_load=True):

    if try_load == True:
        try:
            img = sitk.ReadImage(os.path.join(os.getcwd(), "cache", f'{name}_deconv_texture.nrrd'))
            img = sitk.Resample(img,mask, sitk.Transform(), sitk.sitkLinear, 0, img.GetPixelID())
            return np.clip(sitk.GetArrayFromImage(img), a_min=None, a_max=6)
        except: 
            print("Failed to load texture image, calculating from scratch...")
            pass
    print("Extracting CT texture feature image... (this may take a while)")    
    params = os.path.join(os.getcwd(), "params_ct_deconv.yaml")   #no shape
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params=params, voxelBatch=2500, maskedKernel=True, kernelRadius=3)
    extractor.enabledFeatures = {"glrlm": ["LongRunEmphasis"]}
    result = extractor.execute(img, mask, voxelBased=True)  
    for key, val in six.iteritems(result):
        if isinstance(val, sitk.Image):  # Feature map
            sitk.WriteImage(val, os.path.join(os.getcwd(), "cache", f'{name}_deconv_texture.nrrd'), False)
            img = sitk.ReadImage(os.path.join(os.getcwd(), "cache", f'{name}_deconv_texture.nrrd'))
            img = sitk.Resample(img,mask, sitk.Transform(), sitk.sitkLinear, 0, img.GetPixelID())
            return np.clip(sitk.GetArrayFromImage(img), a_min=None, a_max=6)

 


def blind_deconvolution(pet_img,ct_tex, patient, load_models=False, load_kernel=False, save_dir=None, plot=False):
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "models")

    #first normalize the images 
    max_pet = np.amax(pet_img)
    max_ct = np.amax(ct_tex)
    min_ct = np.amin(ct_tex)

    pet_img_np = (pet_img / max_pet)
    ct_tex_np = ((ct_tex - min_ct) / (max_ct - min_ct))
    ct_tex = np_to_torch(ct_tex_np).unsqueeze(0)
    pet_img = np_to_torch(pet_img_np).unsqueeze(0)

    print(patient)
    print(save_dir)

    lr = 0.001
    lr_kernel = 8e-4
    num_iters=5000

    #put on gpu if possible
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    Gx = model_deconv.Gx(num_input_channels=1, 
                           num_output_channels=1, 
                           upsample_mode='trilinear', 
                           need_sigmoid=True, need_bias=True,act_fun='LeakyReLU').float()
    Gk = model_deconv.Gk(3179, 3179).float().to(device)
    np.random.seed(0)
    Gx_in = torch.from_numpy(np.random.uniform(0,1,(1,1,pet_img.shape[2], pet_img.shape[3], pet_img.shape[4]))).float()
    #Gx_in = torch.from_numpy(np.zeros((1,1,pet_img.shape[2], pet_img.shape[3], pet_img.shape[4]))).float()
    #Gx_in = pet_img.unsqueeze(0)
    Gx.to(device)
    Gx_in = Gx_in.to(device)   
    pet_img = pet_img.to(device)
    


    if load_models:   #if trying to load previous model

        try:
            Gx_and_opt = torch.load(os.path.join(os.getcwd(),"models", f"{patient}_Gx_with_optimizer"))
            Gk_and_opt = torch.load(os.path.join(os.getcwd(),"models", f"{patient}_Gk_with_optimizer"))
            Gx.load_state_dict(Gx_and_opt['model_state_dict'])
            Gk.load_state_dict(Gk_and_opt['model_state_dict'])
            optimizer.load_state_dict(Gx_and_opt['optimizer_state_dict'])
            with open(os.path.join(os.getcwd(),"models",  f"{patient}_model_stuff.txt"), "rb") as fp:
                Gx_in, Gk_in, _,_,_,_, loss_history, initial_step = pickle.load(fp)
            print("Loaded state dicts and loss history. Continuing Training")
        except:
            print("Could not load state dictionaries for model/optimizer. Starting training from scratch...")
    
        print("Beginning blind deconvolution optimization...") 
        for step in range(initial_step, num_iters):
            optimizer.zero_grad()

            out_x = Gx(Gx_in).float()
            out_k = Gk(Gk_in).view(-1,1, 15,15,15).float()

            #sum = torch.sum(out_k)
            #print(sum)
            prediction = nn.functional.conv3d(out_x, out_k, padding=(5,9,9), bias=None).float()

            loss = loss_func(prediction, pet_img, out_k, out_x, step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
            loss_history.append(loss.item())#, fidelity_term.item(), norm_term.item(), tv_loss.item(),tv_loss_mask.item(), loss_kernel.item()])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 10 == 0:
                print(f"""Finished step {step}, loss = {round(loss.item(),8)}""")#, fidelity = {round(fidelity_term.item(),6)}, norm = {round(norm_term.item(),6)}, tv loss = {round(tv_loss.item(),6)}, tv mask loss = {round(tv_loss_mask.item(),6)}, kernel tv loss = {(round(loss_kernel.item(),6))}""")

            if step != 0 and step % 1000 == 0: 
                prediction = prediction.squeeze().detach().cpu().numpy()
                pet_img_view = pet_img.squeeze().detach().cpu().numpy()
                out_k = out_k.squeeze().detach().cpu().numpy()
                out_x = out_x.squeeze().detach().cpu().numpy()
                # plot_3d_image(pet_img_crop_view)
                # plot_3d_image(out_k)
                # plot_3d_image(out_x)
                # plot_3d_image(prediction)

                with open(os.path.join(save_dir,  f"{patient}_model_stuff.txt"), "wb") as fp:
                    pickle.dump([Gx_in, Gk_in, out_x, out_k, pet_img_view, prediction, loss_history, step], fp)
                torch.save({'model_state_dict': Gx.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gx_with_optimizer"))  
                torch.save({'model_state_dict': Gk.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir,f"{patient}_Gk_with_optimizer"))     

        prediction = prediction.squeeze().detach().cpu().numpy()
        pet_img_view = pet_img.squeeze().detach().cpu().numpy()
        out_k = out_k.squeeze().detach().cpu().numpy()
        out_x = out_x.squeeze().detach().cpu().numpy()
        with open(os.path.join(save_dir,  f"{patient}_model_stuff.txt"), "wb") as fp:
            pickle.dump([Gx_in, Gk_in, out_x, out_k, pet_img_view, prediction, loss_history, step], fp)
        torch.save({'model_state_dict': Gx.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gx_with_optimizer"))  
        torch.save({'model_state_dict': Gk.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gk_with_optimizer"))    


    else:
        print("Beginning blind deconvolution optimization...") 
        #plot_3d_image(out_k.squeeze().detach().cpu().numpy())  
        #do the first down-sampled loop
        ####################################################################################################
        small_Gk_in = F.softmax(torch.log(torch.abs(torch.from_numpy(generate_target_kernel(kernel_size=7)).view(-1)+1e-9)),dim=0).float().to(device)  
        small_img = zoom(Gx_in.detach().squeeze().cpu().numpy(), zoom=(0.5, 0.5, 0.5), order=1)
        small_img = np_to_torch(small_img).unsqueeze(0).to(device)
        Gk_small = model_deconv.Gk(7**3, 7**3).float().to(device)
        target = np_to_torch(zoom(pet_img_np, zoom=(0.5,0.5,0.5), order=1)).unsqueeze(0).to(device)
        ct_tex_small = np_to_torch(zoom(ct_tex_np, zoom=(1,1,1), order=1)).unsqueeze(0).to(device)
        optimizer_small = torch.optim.Adam([{'params': Gx.parameters()}, {'params': Gk_small.parameters(), 'lr': lr_kernel}], lr=lr)
        loss_func_small = loss_functions.Deconv_Loss_Small(tv_weight=0.01, ct_weight=0.00001, ct_tex=ct_tex_small, device=device)
        avg_pool = nn.AvgPool3d(kernel_size=(2, 2, 2))
        #we want the kernel model to start off by PREDICTING a gaussian kernel. so we need to have this kernel model pretrained
        kernel_optimizer = torch.optim.Adam([{'params': Gk_small.parameters(), 'lr': lr_kernel}])
        kernel_loss = loss_functions.Kernel_Loss()
        y= deepcopy(small_Gk_in.view(1,1, 7,7,7).float())
        #plot_3d_image(target.detach().squeeze().cpu().numpy())
        #plot_3d_image(generate_target_kernel(kernel_size=7))
        #plot_3d_image(y.cpu().detach().squeeze().numpy())
        Gx_optimizer = torch.optim.Adam([{'params': Gx.parameters(), 'lr': 0.002}])
        Gx_loss = loss_functions.Kernel_Loss()
        pre_loss = []
        #pre train the Gx model to predict last one too 
        for pre_train_step in range(300):
            Gx_optimizer.zero_grad()
            out_x = Gx(small_img).float()
            loss = Gx_loss(F.interpolate(out_x, scale_factor=0.5, mode='trilinear'), target)
            loss.backward()
            Gx_optimizer.step()
            # if pre_train_step % 10 == 0:
            #     print(f"step: {pre_train_step} - Gx loss: {round(loss.item(),9)}")

        # #print("predicted Kernel 2 - pre-training")    
        #plot_3d_image(out_x.squeeze().detach().cpu().numpy())

        for pre_train_step in range(100):
            kernel_optimizer.zero_grad()
            out_k = Gk_small(small_Gk_in).view(1,1, 7,7,7).float()
            loss = kernel_loss(out_k, y)
            loss.backward()
            kernel_optimizer.step()
        start_time = time.time()    
        for small_step in range(1000):
            optimizer_small.zero_grad()
            out_x = Gx(small_img).float()
            out_k = Gk_small(small_Gk_in).view(1,1, 7,7,7).float()
            prediction_interp = F.interpolate(out_x, scale_factor=0.5, mode='trilinear')
            prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(3,3,3), bias=None).float()
            loss = loss_func_small(prediction, target, out_k, prediction_interp, small_step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
            loss.backward()
            pre_loss.append(loss.item())
            optimizer_small.step()

            if small_step % 100 == 0:
                print(f"""Finished step {small_step}, loss = {round(loss.item(),10)}""")#, fidelity = {round(fidelity_term.item(),6)}, nor
                print(f"lr: {optimizer_small.param_groups[0]['lr']}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        mins = int(elapsed_time / 60)
        secs = int(elapsed_time % 60)
        print(f"Elapsed time: {mins} minutes and {secs} seconds")
        #do the second down-sampled loop
        #########################################################################################################
        small_Gk_in = out_k.squeeze().cpu().detach().numpy()
        print("predicted Kernel 1")  
        if plot ==True: 
            plot_3d_image(small_Gk_in)  
        small_Gk_in = zoom(small_Gk_in, zoom=[11/7,11/7,11/7], order=1)
        small_Gk_in = F.softmax(torch.log(torch.abs(np_to_torch(small_Gk_in)).view(-1)+ 1e-9), dim=-1).float().to(device)    
        
        if plot ==True: 
            print("Kernel 1 zoomed")   
            plot_3d_image(torch.reshape(small_Gk_in,(11,11,11)).detach().cpu().numpy())
        
        small_img = out_x.detach().squeeze().cpu().numpy()
        if plot ==True: 
            print("image 1")   
            plot_3d_image(small_img)
        small_img = zoom(small_img, zoom=(2**-0.5,2**-0.5,2**-0.5), order=1)
        if plot ==True: 
            print("image 1 interp")   
            plot_3d_image(out_x.detach().squeeze().cpu().numpy())

        small_img = np_to_torch(small_img).unsqueeze(0).to(device)
        Gk_small = model_deconv.Gk(11**3, 11**3).float().to(device)
        target = np_to_torch(zoom(pet_img_np, zoom=(2**-0.5,2**-0.5,2**-0.5), order=1)).unsqueeze(0).to(device)
        ct_tex_small = np_to_torch(zoom(ct_tex_np, zoom=(2**-0.5,2**-0.5,2**-0.5), order=3)).unsqueeze(0).to(device)
        optimizer_small = torch.optim.Adam([{'params': Gx.parameters()}, {'params': Gk_small.parameters(), 'lr': lr_kernel}], lr=0.001)
        loss_func_small = loss_functions.Deconv_Loss_Small(tv_weight=0.01, ct_weight=0.00001, ct_tex=ct_tex_small, device=device)
        #we want the kernel model to start off by PREDICTING a gaussian kernel. so we need to have this kernel model pretrained
        kernel_optimizer = torch.optim.Adam([{'params': Gk_small.parameters(), 'lr': lr_kernel}])
        kernel_loss = loss_functions.Kernel_Loss()
        y= deepcopy(small_Gk_in.view(-1,1, 11,11,11).float())

        for pre_train_step in range(100):
            kernel_optimizer.zero_grad()
            out_k = Gk_small(small_Gk_in.cuda()).view(-1,1, 11,11,11).float()
            loss = kernel_loss(out_k, y)
            loss.backward()
            kernel_optimizer.step()
            if pre_train_step % 100 == 0:
                print(f"step: {pre_train_step} - Kernel loss: {round(loss.item(),10)}")

        Gx_optimizer = torch.optim.Adam([{'params': Gx.parameters(), 'lr': 0.002}])
        Gx_loss = nn.MSELoss()
        print("predicted Kernel 2 - pre-training")    
        #plot_3d_image(out_k.squeeze().detach().cpu().numpy())
        #pre train the Gx model to predict last one too 
        for pre_train_step in range(300):
            Gx_optimizer.zero_grad()
            out_x = Gx(small_img).float()
            out_x = F.interpolate(out_x, scale_factor=0.5, mode='trilinear')
            loss = Gx_loss(out_x, small_img)
            loss.backward()
            Gx_optimizer.step()
            # if pre_train_step % 100 == 0:
            #     print(f"step: {pre_train_step} - Gx loss: {round(loss.item(),9)}")
        #print("predicted Image 2 - pre-training")    
        #plot_3d_image(out_x.squeeze().detach().cpu().numpy())

        for small_step_2 in range(1000):
            optimizer_small.zero_grad()
            out_x = Gx(small_img).float()
            out_k = Gk_small(small_Gk_in).view(-1,1, 11,11,11).float()

            #sum = torch.sum(out_k)
            #print(sum)
            #prediction = nn.functional.conv3d(out_x, out_k, padding=(5,5,5), bias=None).float()
            #prediction = nn.functional.conv3d(out_x, out_k, padding=(5,5,5), bias=None).float()
            prediction_interp = F.interpolate(out_x, scale_factor=0.5, mode='trilinear')
            prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(5,5,5), bias=None).float()
            loss = loss_func_small(prediction, target, out_k, prediction_interp, small_step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
            # prediction_interp = F.interpolate(out_x, scale_factor=0.5, mode='trilinear')
            # prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(5,5,5), bias=None).float()

            # loss = loss_func_small(prediction, target, out_k, prediction_interp, small_step_2) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
            loss.backward()
            pre_loss.append(loss.item())
            optimizer_small.step()
            if small_step_2 % 100 == 0:
                print(f"""Finished step {small_step_2}, loss = {round(loss.item(),8)}""")#, fidelity = {round(fidelity_term.item(),6)}, nor
                print(f"lr: {optimizer_small.param_groups[0]['lr']}")
        
        with open(os.path.join(save_dir, f"{patient}_pre_loss.txt"), "wb") as fp:
            pickle.dump(pre_loss, fp)
        
        #now upsample the kernel and image back to original size and train models to start with that.
        #############################################################################################################################
        Gk_in = out_k.squeeze().cpu().detach().numpy()
        if plot ==True: 
            print("Final starting kernel")
            plot_3d_image(Gk_in)
        Gk_in = zoom(Gk_in, zoom=[15/11,15/11,15/11], order=1)
        Gk_in = F.softmax(torch.log(torch.abs(np_to_torch(Gk_in).view(-1))+1e-9),dim=-1).float().to(device)  #from 11 to 15
        if plot ==True: 
            print("Final starting kernel - zoomed")
            plot_3d_image(torch.reshape(Gk_in,(15,15,15)).detach().cpu().numpy())
        
        if plot ==True: 
            print("image 1 interp")   
            plot_3d_image(out_x.detach().squeeze().cpu().numpy())

        Gx_in = out_x.detach().squeeze().cpu().numpy()
        if plot ==True: 
            print("Final starting image")
            plot_3d_image(Gx_in)
        Gx_in = zoom(Gx_in, zoom=[pet_img.shape[2]/out_x.shape[2],pet_img.shape[3]/out_x.shape[3],pet_img.shape[4]/out_x.shape[4]], order=1)
        Gx_in = np_to_torch(Gx_in).unsqueeze(0).to(device)

        #delete old model and clear cache
        torch.cuda.empty_cache()
        del Gk_small
        Gk = model_deconv.Gk(15**3, 15**3).float().to(device)

        #we want the kernel model to start off by PREDICTING a gaussian kernel. so we need to have this kernel model pretrained
        kernel_optimizer = torch.optim.Adam([{'params': Gk.parameters(), 'lr': lr_kernel}])
        kernel_loss = loss_functions.Kernel_Loss()
        y= deepcopy(Gk_in.view(-1,1, 15,15,15).float())
        print("final kernel y")
        #plot_3d_image(y.squeeze().detach().cpu().numpy())
        for pre_train_step in range(100):
            kernel_optimizer.zero_grad()
            out_k = Gk(Gk_in.cuda()).view(-1,1, 15,15,15).float()
            loss = kernel_loss(out_k, y)
            loss.backward()
            kernel_optimizer.step()
            # if pre_train_step % 10 ==0 :
            #     print(f"Step: {pre_train_step} | Kernel loss: {round(loss.item(),9)}")


        Gx_optimizer = torch.optim.Adam([{'params': Gx.parameters(), 'lr': 0.002}])
        Gx_loss = loss_functions.Kernel_Loss()
        #pre train the Gx model to predict last one too 
        for pre_train_step in range(300):
            Gx_optimizer.zero_grad()
            out_x = F.interpolate(Gx(Gx_in), scale_factor=0.5, mode='trilinear')
            loss = Gx_loss(out_x, Gx_in)
            loss.backward()
            Gx_optimizer.step()
            # if pre_train_step % 10 == 0:
            #     print(f"step: {pre_train_step} - Gx loss: {round(loss.item(),9)}")


        ct_tex = np_to_torch(zoom(ct_tex_np, zoom=(2,2,2), order=1)).unsqueeze(0).to(device)  
        lr = 0.001  
        torch.cuda.empty_cache()
        optimizer = torch.optim.Adam([{'params': Gx.parameters()}, {'params': Gk.parameters(), 'lr': lr_kernel}], lr)
        loss_func = loss_functions.Deconv_Loss(tv_weight=0.01, ct_weight=0.00001, ct_tex=ct_tex.to(device), device=device)
        scheduler = MultiStepLR(optimizer, milestones=[2000, 4000], gamma=0.5)
        loss_history = []
        initial_step = 0
        
        start_time = time.time()
        for step in range(initial_step, num_iters):
            optimizer.zero_grad()
            out_x = Gx(Gx_in).float()
            out_k = Gk(Gk_in).view(-1,1, 15,15,15).float()

            prediction_interp = F.interpolate(out_x, scale_factor=0.5, mode='trilinear')
            prediction = nn.functional.conv3d(prediction_interp, out_k, padding=(7,7,7), bias=None).float()
            loss = loss_func(prediction, pet_img, out_k, out_x, step) #, fidelity_term, norm_term, tv_loss, tv_loss_mask, loss_kernel
            loss_history.append(loss.item())#, fidelity_term.item(), norm_term.item(), tv_loss.item(),tv_loss_mask.item(), loss_kernel.item()])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                print(f"""Finished step {step}, loss = {round(loss.item(),8)}""")#, fidelity = {round(fidelity_term.item(),6)}, norm = {round(norm_term.item(),6)}, tv loss = {round(tv_loss.item(),6)}, tv mask loss = {round(tv_loss_mask.item(),6)}, kernel tv loss = {(round(loss_kernel.item(),6))}""")
                print(f"lr: {optimizer.param_groups[0]['lr']}")
            if step != 0 and step % 1000 == 0: 
                #print the time taken 
                end_time = time.time()
                elapsed_time = end_time - start_time
                mins = int(elapsed_time / 60)
                secs = int(elapsed_time % 60)
                
                print(f"Elapsed time: {mins} minutes and {secs} seconds")
                
                prediction = prediction.squeeze().detach().cpu().numpy()
                pet_img_view = pet_img.squeeze().detach().cpu().numpy()
                out_k = out_k.squeeze().detach().cpu().numpy()
                out_x = out_x.squeeze().detach().cpu().numpy()
                prediction_interp = prediction_interp.squeeze().detach().cpu().numpy()
                if plot == True:
                    plot_3d_image(pet_img_view)
                    plot_3d_image(out_k)
                    plot_3d_image(out_x)
                    plot_3d_image(prediction)
                    plot_3d_image(prediction_interp)
                
                with open(os.path.join(save_dir,  f"{patient}_model_stuff.txt"), "wb") as fp:
                    pickle.dump([out_x*max_pet, out_k, pet_img_view*max_pet, prediction*max_pet, ct_tex.squeeze().detach().cpu().numpy(), loss_history, step], fp)
                torch.save({'model_state_dict': Gx.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gx_with_optimizer.pt"))  
                torch.save({'model_state_dict': Gk.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gk_with_optimizer.pt"))     

                start_time = time.time()

        prediction = prediction.squeeze().detach().cpu().numpy()
        pet_img_view = pet_img.squeeze().detach().cpu().numpy()
        out_k = out_k.squeeze().detach().cpu().numpy()
        out_x = out_x.squeeze().detach().cpu().numpy()
        with open(os.path.join(save_dir,  f"{patient}_model_stuff.txt"), "wb") as fp:
            pickle.dump([out_x*max_pet, out_k, pet_img_view*max_pet, prediction*max_pet, ct_tex.squeeze().detach().cpu().numpy(), loss_history, step], fp)
        torch.save({'model_state_dict': Gx.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gx_with_optimizer.pt"))  
        torch.save({'model_state_dict': Gk.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, os.path.join(save_dir, f"{patient}_Gk_with_optimizer.pt"))    
    return prediction * max_pet, out_x * max_pet, out_k
    
if __name__ == "__main__":
    main()

