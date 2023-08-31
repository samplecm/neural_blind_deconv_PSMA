from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
import numpy as np
import copy
from copy import deepcopy
import scipy
import matplotlib.pyplot as plt

import os
import pickle
from skimage import morphology, measure
from matplotlib.widgets import Slider, Button
from scipy.ndimage import binary_dilation, binary_erosion, convolve
import bisect
from utils import *


def get_all_structure_masks(img_series, structures_dict, segmentation_types=["reg", "si","ap","ml"], filled=True, img_type="reg"):
    if img_type == "reg":
        coords_array = img_series.coords_array_img
    elif img_type == "deblur":
        coords_array = img_series.deconv_coords_2x_img
    for structure_name in structures_dict:

        contours = structures_dict[structure_name].whole_roi_img
  
        mask_array = get_contour_masks(contours, coords_array, filled=filled)
        if img_type == "reg":
            structures_dict[structure_name].whole_roi_masks = mask_array
        elif img_type == "deblur":
            structures_dict[structure_name].whole_roi_masks_deblur = mask_array    
        for segmentation_type in segmentation_types:
            if img_type == "reg":
                if not hasattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type):
                    continue
                #now get subsegment masks
                contours = getattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type)
                subseg_masks = []
                for subseg in contours:
                    mask_array = get_contour_masks(subseg, coords_array)
                    subseg_masks.append(mask_array)
                setattr(structures_dict[structure_name], f"subseg_masks_{segmentation_type}", deepcopy(subseg_masks))   
                #structures_dict[structure_name].subseg_masks = deepcopy(subseg_masks)    
            elif img_type == "deblur":
                if not hasattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type):
                    continue
                #now get subsegment masks
                contours = getattr(structures_dict[structure_name], "segmented_contours_" + segmentation_type)
                subseg_masks = []
                for subseg in contours:
                    mask_array = get_contour_masks(subseg, coords_array)
                    subseg_masks.append(mask_array)
                setattr(structures_dict[structure_name], f"subseg_masks_{segmentation_type}_deblur", deepcopy(subseg_masks))   
                #structures_dict[structure_name].subseg_masks = deepcopy(subseg_masks)       
    return structures_dict

def get_contour_masks(contours, array, filled=True):
    
    num_slices, len_y, len_x = array.shape[1:]
    contour_masks = np.zeros_like(array[0,:])
    contours = cartesian_to_pixel_coordinates(clone_list(contours), array)
    if filled == True:
        fill = 1
    else:
        fill = 0 
    #first get a list of z values with slices on them, and list of those slice images.
    mask_list = []
    z_list = []

    
    for contour in contours:
        contour_mask_filled = Image.new('L', (len_x, len_y), 0)        
        if contour == []:
            continue
        for slice in contour:
            if len(slice) < 3:
                continue
            contourPoints = []
            for point in slice:
                contourPoints.append((int(point[0]), int(point[1]))) #changed
            ImageDraw.Draw(contour_mask_filled).polygon(contourPoints, outline= 1, fill = fill)   
            mask_list.append(np.array(contour_mask_filled).astype(np.float32))   
            z_list.append(slice[0][2])
            break
    #now go through image slices and interpolate mask slices 
    for idx in range(num_slices):
        img_z = array[2,idx, 0,0]
        closest_slices = find_closest_slices(z_list, img_z, 0.5)
        if closest_slices is None:
            continue #slice is 0
        if type(closest_slices) == int:
            contour_masks[idx, :,:] = mask_list[closest_slices]
        elif type(closest_slices) == tuple:
            #need to interpolate between slices
            slice_1 = mask_list[closest_slices[0]]
            slice_2 = mask_list[closest_slices[1]]
            weight_1 = 1 - (img_z - z_list[closest_slices[0]])  /  (z_list[closest_slices[1]]- z_list[closest_slices[0]])
            weight_2 = 1 - weight_1
            interp_slice = slice_1 * weight_1 + slice_2 * weight_2
            #plot_2d_image(interp_slice)
            interp_slice = convolve(interp_slice, np.ones((2,2))/4)
            #plot_2d_image(interp_slice)
            interp_slice = interp_slice > 0.5
            contour_masks[idx, :, :] = interp_slice.astype(int)   
            # plot_2d_image(slice_1)
            # plot_2d_image(slice_2)
            # plot_2d_image(interp_slice)  
            n1= np.count_nonzero(slice_1)
            n2 = np.count_nonzero(slice_2)
            n3 = np.count_nonzero(interp_slice)

    return contour_masks.astype(np.bool)

def find_closest_slices(sorted_list, target_value, range_value):
    index = bisect.bisect_left(sorted_list, target_value)
    
    if 0 <= index < len(sorted_list):
        closest_value = sorted_list[index]
        if abs(closest_value - target_value) <= range_value:
            return index
        if index > 0:  
            closest_value = sorted_list[index-1]
            if abs(closest_value - target_value) <= range_value:
                return index
            
    if 0 < index < len(sorted_list):
        return index-1, index
    
    return None

def clone_list(list):
    listCopy = copy.deepcopy(list)
    return listCopy  

def cartesian_to_pixel_coordinates(contours, array):
    #convert x and y values for a contour into the pixel indices where they are on the pet array
    xVals = array[0,0,0,:]
    yVals = array[1,0,:,0]
    for contour in contours: 
        if len(contour) == 0:
            continue
        for slice in contour:
            if len(slice) == 0: continue
            for point in slice:
                point[0] = min(range(len(xVals)), key=lambda i: abs(xVals[i]-point[0]))
                point[1] = min(range(len(yVals)), key=lambda i: abs(yVals[i]-point[1]))
    return contours  
