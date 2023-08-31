import torch
import torch.nn as nn
import torchvision
import sys
import cv2
import numpy as np
from PIL import Image
import PIL
import math
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import erf
from copy import deepcopy
from scipy.ndimage import label, find_objects

import matplotlib.pyplot as plt
import random

def remove_small_islands(mask, min_size):
    # Label connected components in the binary mask
    labeled_mask, num_features = label(mask)
    
    # Find the sizes of the labeled components
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # Exclude the background
    
    # Find the components to keep (larger than min_size)
    valid_components = np.where(component_sizes >= min_size)[0]
    
    # Create a new mask without the small islands
    cleaned_mask = np.isin(labeled_mask, valid_components).astype(np.uint8)
    
    return cleaned_mask

def plot_2d_image(image):
    masked_img = deepcopy(image)
    vmin = np.nanmin(masked_img)
    vmax = np.nanmax(masked_img)
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(image[:,:], cmap='plasma', vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(im, ax=ax)
    plt.show(block=True)
    plt.close("all")
    return

def plot_3d_voxel_image(image):

    cmap = plt.get_cmap('viridis')
    masked_image = np.ma.masked_invalid(deepcopy(image))   #dont let nan points have color
    

    norm = plt.Normalize(vmin=np.min(masked_image), vmax=np.max(masked_image))
    colors = cmap(norm(masked_image))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # Set the background color to white
    ax.set_facecolor((1.0, 1.0, 1.0))

    # Remove axis grids
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    #ax.voxels(masked_image, facecolors=colors, edgecolor=None, shade=False)
    # Get the dimensions of the voxel array
    depth, height, width = masked_image.shape
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)
    # Loop through each voxel and plot it as a solid cube
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if not image[z, y, x] == 0:
                    # Define the vertices of the cube
                    vertices = [
                        (x, y, z),
                        (x + 1, y, z),
                        (x + 1, y + 1, z),
                        (x, y + 1, z),
                        (x, y, z + 1),
                        (x + 1, y, z + 1),
                        (x + 1, y + 1, z + 1),
                        (x, y + 1, z + 1)
                    ]

                    # Define the faces of the cube using the vertices
                    faces = [
                        [vertices[0], vertices[1], vertices[2], vertices[3]],
                        [vertices[4], vertices[5], vertices[6], vertices[7]],
                        [vertices[0], vertices[1], vertices[5], vertices[4]],
                        [vertices[2], vertices[3], vertices[7], vertices[6]],
                        [vertices[0], vertices[3], vertices[7], vertices[4]],
                        [vertices[1], vertices[2], vertices[6], vertices[5]]
                    ]

                    # Create a Poly3DCollection object to represent the voxel
                    voxel = Poly3DCollection(faces, facecolors=[colors[z, y, x]], edgecolor='none')

                    # Add the voxel to the plot
                    ax.add_collection3d(voxel)
    plt.show(block=True)
    return 
def plot_3d_kernel(image):
    image[image < 0.01] = np.nan
    alpha_min = 0.2
    alpha_max = 1
    cmap = plt.get_cmap('viridis')
    masked_image = np.ma.masked_invalid(deepcopy(image))   #dont let nan points have color
    

    norm = plt.Normalize(vmin=np.min(masked_image), vmax=np.max(masked_image))
    colors = cmap(norm(masked_image))

    #make smaller colors more transparent
    alpha_values = (masked_image - np.max(masked_image)) #/ (np.max(masked_image) - np.min(masked_image))
    alpha_values = np.exp(300*alpha_values)#alpha_min + alpha_values * (alpha_max - alpha_min)

    # Apply the alpha values to the colors
    colors[..., -1] = alpha_values

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # Set the background color to white
    ax.set_facecolor((1.0, 1.0, 1.0))

    # Remove axis grids
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    #ax.voxels(masked_image, facecolors=colors, edgecolor=None, shade=False)
    # Get the dimensions of the voxel array
    depth, height, width = masked_image.shape
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)
    # Loop through each voxel and plot it as a solid cube
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if not np.ma.is_masked(masked_image[z, y, x]):
                    # Define the vertices of the cube
                    vertices = [
                        (x, y, z),
                        (x + 1, y, z),
                        (x + 1, y + 1, z),
                        (x, y + 1, z),
                        (x, y, z + 1),
                        (x + 1, y, z + 1),
                        (x + 1, y + 1, z + 1),
                        (x, y + 1, z + 1)
                    ]

                    # Define the faces of the cube using the vertices
                    faces = [
                        [vertices[0], vertices[1], vertices[2], vertices[3]],
                        [vertices[4], vertices[5], vertices[6], vertices[7]],
                        [vertices[0], vertices[1], vertices[5], vertices[4]],
                        [vertices[2], vertices[3], vertices[7], vertices[6]],
                        [vertices[0], vertices[3], vertices[7], vertices[4]],
                        [vertices[1], vertices[2], vertices[6], vertices[5]]
                    ]

                    # Create a Poly3DCollection object to represent the voxel
                    voxel = Poly3DCollection(faces, facecolors=[colors[z, y, x]], edgecolor='none')

                    # Add the voxel to the plot
                    ax.add_collection3d(voxel)
    plt.show(block=True)
    return 
def plot_3d_image(image, view="axial"):

    if view=='axial':
        def update(val):
            slice_index = int(slider.val)
            ax.imshow(image[slice_index, :,:], cmap='plasma', vmin=np.nanmin(image[val,:,:]), vmax=np.nanmax(image[val,:,:]))
            fig.canvas.draw_idle()
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(image[int(image.shape[0]/2), :,:], cmap='plasma', vmin=np.nanmin(image[int(image.shape[0]/2), :,:]), vmax=np.nanmax(image[int(image.shape[0]/2), :,:]))
        colorbar = plt.colorbar(im, ax=ax)
        ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
        slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=image.shape[0]-1, valstep=1, valinit=int(image.shape[0]/2))
        slider.on_changed(update)
        plt.show(block=True)
        plt.close("all")
        return
    elif view=='cor':
        def update(val):
            slice_index = int(slider.val)
            ax.imshow(image[:, slice_index,:], cmap='plasma', vmin=np.amin(image), vmax=np.amax(image), origin='lower')
            fig.canvas.draw_idle()
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(image[:, int(image.shape[1]/2),:], cmap='plasma', vmin=np.amin(image), vmax=np.amax(image), origin='lower')
        colorbar = plt.colorbar(im, ax=ax)
        ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
        slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=image.shape[1]-1, valstep=1, valinit=int(image.shape[1]/2))
        slider.on_changed(update)
        plt.show(block=True)
        plt.close("all")
        return

    if view=='sag':
        def update(val):
            slice_index = int(slider.val)
            ax.imshow(image[:, :,slice_index], cmap='plasma', vmin=np.amin(image), vmax=np.amax(image), origin='lower')
            fig.canvas.draw_idle()
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(image[:, :,int(image.shape[2]/2)], cmap='plasma', vmin=np.amin(image), vmax=np.amax(image), origin='lower')
        colorbar = plt.colorbar(im, ax=ax)
        ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
        slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=image.shape[2]-1, valstep=1, valinit=int(image.shape[2]/2))
        slider.on_changed(update)
        plt.show(block=True)
        plt.close("all")
        return


def generate_target_kernel(kernel_size=15, peak_width=2, kernel_type="gaussian", stretch_factors=[1,1,1],skew_direction="+x", alpha=2):
    #specify the std of the kernel with peak_width, and whether you want a gaussian, or a skewed curve. if kernel_type="skew", you 
    #must also specify skew_direction as "+{x,y,z}" or "-{x,y,z}" and 
    # Create an empty target kernel tensor
    
    target_kernel = np.zeros((kernel_size,kernel_size,kernel_size))

    # Calculate the coordinates of the center pixel
    center_coords = [dim // 2 for dim in target_kernel.shape]
    if kernel_type == "gaussian":
        # Generate a Gaussian distribution
        for z in range(kernel_size):
            for y in range(kernel_size):
                for x in range(kernel_size):
                    # Calculate the distance from the center pixel
                    dist_z = (z - center_coords[0]) *stretch_factors[0]
                    dist_y = (y - center_coords[1]) * stretch_factors[1]
                    dist_x = (x - center_coords[2]) * stretch_factors[2]
                    dist = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)

                    # Calculate the value of the Gaussian distribution
                    target_kernel[z, y, x] = np.exp(-(dist)**2 / (2 * peak_width**2))

    elif kernel_type == "skew":
        skew_axis = skew_direction[1] 
        if skew_direction[0] == "-":
            alpha *= -1

        for z in range(kernel_size):
            for y in range(kernel_size):
                for x in range(kernel_size):
                    # Calculate the distance from the center pixel
                    dist_z = (z - center_coords[0])  
                    dist_y = (y - center_coords[1])
                    dist_x = (x - center_coords[2])
                    dist = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
                    if skew_axis == "x":
                        dist_skew = dist_x
                    elif skew_axis == "y":
                        dist_skew = dist_y
                    elif skew_axis == "z":
                        dist_skew = dist_z        
                    # Calculate the value of the Gaussian distribution
                    target_kernel[z, y, x] = np.exp(-(dist)**2 / (2 * peak_width**2)) * (1 + erf(alpha * dist_skew))

    # Normalize the target kernel
    target_kernel /= np.sum(target_kernel)
    # plot_3d_image(target_kernel)

    return target_kernel        



        

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''
    imgsize = img.shape

    new_size = (imgsize[0] - imgsize[0] % d,
                imgsize[1] - imgsize[1] % d)

    bbox = [
            int((imgsize[0] - new_size[0])/2),
            int((imgsize[1] - new_size[1])/2),
            int((imgsize[0] + new_size[0])/2),
            int((imgsize[1] + new_size[1])/2),
    ]

    img_cropped = img[0:new_size[0],0:new_size[1],:]
    return img_cropped


def np_to_torch(img_np):
    
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):

    return img_var.detach().cpu().numpy()[0]
