import pydicom 
import radiomics
import os
from astropy.io import fits
import six
import glob
import numpy as np

import nrrd
import argparse
import re

import datetime
from scipy.fftpack import fftn, ifftn
import copy
import pickle


class Img_Series:
    def __init__(self, data_dir, modality, img_type=None, load_method=None, patient_num=None):
        if img_type is None: #pet
            print(f"Collecting {modality} images")
            #object will store the imgs as one 4d array, where 4th dimension is for storing cartesian corrdinates of every pixel. 
            metadata_list = get_img_series_metadata(data_dir, modality=modality)
            iop = metadata_list[0].ImageOrientationPatient
            self.name = str(metadata_list[0].PatientName)
            self.acquisition_date = str(metadata_list[0].AcquisitionDate)
            self.modality = modality
        # we will refer to patient cartesian coordinates as x,y,z and the image coordinates as a,b,c
            self.cos_ax = float(iop[0])
            self.cos_ay = float(iop[1])
            self.cos_az = float(iop[2])
            self.cos_bx = float(iop[3])
            self.cos_by = float(iop[4])
            self.cos_bz = float(iop[5])

            if self.cos_ay != 0.0 or self.cos_az != 0.0 or self.cos_bx != 0.0 or self.cos_bz != 0.0:
                raise Exception("Need to handle rotated reference frame.")

            #get cosines with c by taking cross product. 
            cross_prod = np.cross(np.array(iop[0:3]), np.array(iop[3:]))
            self.cos_cx = cross_prod[0]
            self.cos_cy = cross_prod[1]
            self.cos_cz = cross_prod[2]

            self.slice_thickness = np.linalg.norm(np.array(metadata_list[1].ImagePositionPatient) - np.array(metadata_list[0].ImagePositionPatient)) #can't always just use the slice thickness metadata attribute (not always same as slice separation, and slice separation attr not always there.)
            self.pixel_spacing = metadata_list[0].PixelSpacing
            pixel_spacing_list = [metadata_list[i].PixelSpacing for i in range(100,len(metadata_list))]
            slice_thickness_list = [np.linalg.norm(np.array(metadata_list[i+1].ImagePositionPatient) - np.array(metadata_list[i].ImagePositionPatient)) for i in range(100,len(metadata_list)-1)]
            self.origin = metadata_list[0].ImagePositionPatient

            x_orig = float(self.origin[0])
            y_orig = float(self.origin[1])
            z_orig = float(self.origin[2])
            img_list = []
            ipp_list = []

            if modality.lower() == "ct":
                for data in metadata_list:
                    img_list.append((data.pixel_array * float(data.RescaleSlope)) + float(data.RescaleIntercept))
                    ipp_list.append(data.ImagePositionPatient)
            elif modality.lower() == "pet":    #convert to suvbw.
                half_life = float(metadata_list[0][0x0054, 0x0016][0][0x0018, 0x1075].value)
                rad_start_time = metadata_list[0][0x0054, 0x0016][0][0x0018, 0x1072].value
                rad_start_time = HHMMSS_To_S(rad_start_time)
                total_dose = float(metadata_list[0][0x0054,0x0016][0][0x0018, 0x1074].value)    
                patient_weight = float(metadata_list[0][0x0010, 0x1030].value)
                patient_height = float(metadata_list[0][0x0010, 0x1020].value) * 100 #convert to cm (alwasy in m)
                patient_sex = metadata_list[0][0x0010,0x0040].value
                patient_lbm = get_hume_lbm(patient_weight, patient_height, patient_sex) * 1000 #convert to g
                patient_bsa = 0.007184 * np.power(patient_weight, 0.425)*np.power(patient_height, 0.725) * 10000 #convert to cm^2
                suv_factors_folder = os.path.join(os.getcwd(), "suv_normalization_factors")
                if not os.path.exists(suv_factors_folder):
                    os.mkdir(suv_factors_folder)
                suv_factors_folder = os.path.join(os.getcwd(), "suv_normalization_factors", patient_num)
                if not os.path.exists(suv_factors_folder):
                    os.mkdir(suv_factors_folder)
                suv_factors_file = os.path.join(suv_factors_folder, "factors")
                with open(suv_factors_file, "wb") as fp:
                    pickle.dump([patient_weight * 1000, patient_lbm, patient_bsa], fp)
                self.suv_factors = [patient_weight * 1000, patient_lbm, patient_bsa]       #bw, lbm, bsa

                if not os.path.exists(suv_factors_folder):
                    os.mkdir(suv_factors_folder)
                for data in metadata_list:
                    rescale_slope = float(data[0x0028, 0x1053].value)
                    acquisition_time = data[0x0008,0x0031].value
                    acquisition_time = HHMMSS_To_S(acquisition_time)   
                    total_dose_corrected = float(total_dose * 2 **((rad_start_time-acquisition_time)/half_life))

                    img = data.pixel_array * rescale_slope  / total_dose_corrected


                    
                    img_list.append(img)
                    ipp_list.append(data.ImagePositionPatient)
                #need to convert to suvbw 
            row_count, col_count = img_list[0].shape    
            #will return a shape [3 , (#z) , (#y), (#x)] array where first index has coordinate specifications.
            
            #1 --> x value of each pixel
            #2 --> y value of each pixel 
            #3 --> z value of each pixel 
            #array is sorted from smallest to largest z. 
            img_array = np.zeros([len(img_list), row_count, col_count], dtype=np.float32)   #3 first dimensions are for units in suvbw, suvlbm, suvbsa
            coords_array = np.zeros([3,len(img_list), row_count, col_count], dtype=np.float32)
            coords_array_img = np.zeros([3,len(img_list), row_count, col_count], dtype=np.float32)
            for i,img in enumerate(img_list):
                img_array[i,:,:] = img



            for z_idx in range(len(img_list)):
                corner_x, corner_y, corner_z = ipp_list[z_idx]

                for y_idx in range(row_count):
                    b= y_idx * self.pixel_spacing[0]
                    for x_idx in range(col_count):
                        a = x_idx * self.pixel_spacing[1] 
                        #c = z_idx * self.slice_thickness
                        c = (corner_z - z_orig) * self.cos_cz
                        z = corner_z + (b * self.cos_bz) + (a * self.cos_az)
                        x = corner_x + (b * self.cos_bx) + (a * self.cos_ax)
                        y = corner_y + (b * self.cos_by) + (a * self.cos_ay)
                        # test_a = (x-x_orig) * self.cos_ax + (y-y_orig) * self.cos_ay+ (z-z_orig) * self.cos_az
                        # test_b = (x-x_orig) * self.cos_bx + (y-y_orig) * self.cos_by+ (z-z_orig) * self.cos_bz
                        # test_c = (x-x_orig) * self.cos_cx + (y-y_orig) * self.cos_cy+ (z-z_orig) * self.cos_cz
                        coords_array[0,z_idx,y_idx, x_idx] = x
                        coords_array[1,z_idx,y_idx, x_idx] = y
                        coords_array[2,z_idx,y_idx, x_idx] = z

                        coords_array_img[0,z_idx,y_idx, x_idx] = a
                        coords_array_img[1,z_idx,y_idx, x_idx] = b
                        coords_array_img[2,z_idx,y_idx, x_idx] = c
                print(f"Processed {z_idx+1} of {len(img_list)} image slices")       
            #only save parts of image that will have parotids   
            num_slices = img_array.shape[0]      
            starting_slice = int(img_array.shape[0] * 0.66)
            ending_slice = int(img_array.shape[0] - 20)
            self.coords_array = coords_array[:,starting_slice:ending_slice,:,:]
            self.coords_array_img = coords_array_img[:,starting_slice:ending_slice,:,:]  
            self.image_array = img_array[starting_slice:ending_slice,:,:]          
    
        elif img_type == "diffusion" and load_method == "fits":
            #in this  case, going to make an image object for each b value and store all in a dictionary (one object for each b)

            fits_dir = os.path.join(data_dir, "fits")
            all_b_value_imgs = {}
            for file in os.listdir(fits_dir):
                fits_file = os.path.join(fits_dir, file)
                img_data = fits_opener(fits_file)    #includes list of dict containing data for each slice. 
                all_b_value_imgs[img_data[0]["b_value"]] = img_data
            #now save the data into image object with images sorted in a dictionary by b value, with 3d image for each

            iop = img_data[0]["iop"]
            self.iop = iop
            self.name = str(img_data[0]["patient_name"])
            self.acquisition_date = img_data[0]["acquisition_date"]
            self.modality = "MR"
            self.origin = img_data[0]["ipp"]
            x_orig = float(self.origin[0])
            y_orig = float(self.origin[1])
            z_orig = float(self.origin[2])
        # we will refer to patient cartesian coordinates as x,y,z and the image coordinates as a,b,c
            self.cos_ax = float(iop[0])
            self.cos_ay = float(iop[1])
            self.cos_az = float(iop[2])
            self.cos_bx = float(iop[3])
            self.cos_by = float(iop[4])
            self.cos_bz = float(iop[5])

            #get cosines with c by taking cross product. 
            cross_prod = np.cross(np.array(iop[0:3]), np.array(iop[3:]))
            self.cos_cx = cross_prod[0]
            self.cos_cy = cross_prod[1]
            self.cos_cz = cross_prod[2]

            self.slice_thickness = img_data[0]["slice_thickness"]   #comes from space between slices attr
            self.pixel_spacing = img_data[0]["pixel_spacing"]
            # pixel_spacing_list = [metadata_list[i].PixelSpacing for i in range(100,len(metadata_list))]
            # slice_thickness_list = [np.linalg.norm(np.array(metadata_list[i+1].ImagePositionPatient) - np.array(metadata_list[i].ImagePositionPatient)) for i in range(100,len(metadata_list)-1)]
            self.origin = img_data[0]["ipp"]
            img_list = []
            ipp_list = []

            b_value_img_dict = {} #hold 3d images for each b value
            #sort b values
            b_values = list(all_b_value_imgs.keys())
            b_values.sort()
            row_count, col_count = img_data[0]["image"].shape

            #make a 3d image for eadch b value 
            for b in b_values:
                img_array = np.zeros([len(img_data), row_count, col_count], dtype=np.float32)  
                for i, img_slice in enumerate(all_b_value_imgs[b]):
                    img_array[i,:,:] = img_slice["image"]
                b_value_img_dict[b] = img_array

            #get the coordinates array to know the cartesian coordinates in each voxel center.    
            coords_array = np.zeros([3,len(img_data), row_count, col_count], dtype=np.float32)    #dicom geometry
            coords_array_img = np.zeros([3,len(img_data), row_count, col_count], dtype=np.float32)   #converted to image geometry.
            
            for z_idx in range(len(img_data)):
                corner_x, corner_y, corner_z = img_data[z_idx]["ipp"]

                for y_idx in range(row_count):
                    b= y_idx * self.pixel_spacing[0]
                    for x_idx in range(col_count):
                        a = x_idx * self.pixel_spacing[1] 
                        #c = z_idx * self.slice_thickness
                        c = (corner_z - z_orig) * self.cos_cz
                        z = corner_z + (b * self.cos_bz) + (a * self.cos_az)
                        x = corner_x + (b * self.cos_bx) + (a * self.cos_ax)
                        y = corner_y + (b * self.cos_by) + (a * self.cos_ay)
                        # test_a = (x-x_orig) * self.cos_ax + (y-y_orig) * self.cos_ay+ (z-z_orig) * self.cos_az
                        # test_b = (x-x_orig) * self.cos_bx + (y-y_orig) * self.cos_by+ (z-z_orig) * self.cos_bz
                        # test_c = (x-x_orig) * self.cos_cx + (y-y_orig) * self.cos_cy+ (z-z_orig) * self.cos_cz
                        coords_array[0,z_idx,y_idx, x_idx] = x
                        coords_array[1,z_idx,y_idx, x_idx] = y
                        coords_array[2,z_idx,y_idx, x_idx] = z

                        coords_array_img[0,z_idx,y_idx, x_idx] = a
                        coords_array_img[1,z_idx,y_idx, x_idx] = b
                        coords_array_img[2,z_idx,y_idx, x_idx] = c
                print(f"Processed {z_idx+1} of {len(img_data)} image slices")        


            self.coords_array = coords_array
            self.coords_array_img = coords_array_img
            self.image_arrays = b_value_img_dict 

        elif img_type == "diffusion" and load_method == "adc":
            #in this case, just loading pre-made adc image.
            #in this  case, going to make an image object for each b value and store all in a dictionary (one object for each b)
            fits_dir = os.path.join(data_dir, "fits")

            fits_file = os.path.join(fits_dir, os.listdir(fits_dir)[0])
            img_data = fits_opener_adc(fits_file)    #includes list of dict containing data for each slice. 

            #now save the data into image object with images sorted in a dictionary by b value, with 3d image fore each

            iop = img_data[0]["iop"]
            self.iop = iop
            self.name = str(img_data[0]["patient_name"])
            self.acquisition_date = img_data[0]["acquisition_date"]
            self.modality = "MR"
            self.origin = img_data[0]["ipp"]
            x_orig = float(self.origin[0])
            y_orig = float(self.origin[1])
            z_orig = float(self.origin[2])
        # we will refer to patient cartesian coordinates as x,y,z and the image coordinates as a,b,c
            self.cos_ax = float(iop[0])
            self.cos_ay = float(iop[1])
            self.cos_az = float(iop[2])
            self.cos_bx = float(iop[3])
            self.cos_by = float(iop[4])
            self.cos_bz = float(iop[5])

            #get cosines with c by taking cross product. 
            cross_prod = np.cross(np.array(iop[0:3]), np.array(iop[3:]))
            self.cos_cx = cross_prod[0]
            self.cos_cy = cross_prod[1]
            self.cos_cz = cross_prod[2]

            self.slice_thickness = img_data[0]["slice_thickness"]   #comes from space between slices attr
            self.pixel_spacing = img_data[0]["pixel_spacing"]
            # pixel_spacing_list = [metadata_list[i].PixelSpacing for i in range(100,len(metadata_list))]
            # slice_thickness_list = [np.linalg.norm(np.array(metadata_list[i+1].ImagePositionPatient) - np.array(metadata_list[i].ImagePositionPatient)) for i in range(100,len(metadata_list)-1)]
            self.origin = img_data[0]["ipp"]
            img_list = []
            ipp_list = []

                #need to convert to suvbw 
            row_count, col_count = img_data[0]["image"].shape


            img_array = np.zeros([len(img_data), row_count, col_count], dtype=np.float32)   #3 first dimensions are for units in suvbw, suvlbm, suvbsa
            for i, img_slice in enumerate(img_data):
                img_array[i,:,:] = img_slice["image"]


            #get the coordinates array to know the cartesian coordinates in each voxel center.    
            coords_array = np.zeros([3,len(img_data), row_count, col_count], dtype=np.float32)    #dicom geometry
            coords_array_img = np.zeros([3,len(img_data), row_count, col_count], dtype=np.float32)   #converted to image geometry.
            
            for z_idx in range(len(img_data)):
                corner_x, corner_y, corner_z = img_data[z_idx]["ipp"]

                for y_idx in range(row_count):
                    b= y_idx * self.pixel_spacing[0]
                    for x_idx in range(col_count):
                        a = x_idx * self.pixel_spacing[1] 
                        #c = z_idx * self.slice_thickness
                        c = (corner_z - z_orig) * self.cos_cz
                        z = corner_z + (b * self.cos_bz) + (a * self.cos_az)
                        x = corner_x + (b * self.cos_bx) + (a * self.cos_ax)
                        y = corner_y + (b * self.cos_by) + (a * self.cos_ay)
                        # test_a = (x-x_orig) * self.cos_ax + (y-y_orig) * self.cos_ay+ (z-z_orig) * self.cos_az
                        # test_b = (x-x_orig) * self.cos_bx + (y-y_orig) * self.cos_by+ (z-z_orig) * self.cos_bz
                        # test_c = (x-x_orig) * self.cos_cx + (y-y_orig) * self.cos_cy+ (z-z_orig) * self.cos_cz
                        coords_array[0,z_idx,y_idx, x_idx] = x
                        coords_array[1,z_idx,y_idx, x_idx] = y
                        coords_array[2,z_idx,y_idx, x_idx] = z

                        coords_array_img[0,z_idx,y_idx, x_idx] = a
                        coords_array_img[1,z_idx,y_idx, x_idx] = b
                        coords_array_img[2,z_idx,y_idx, x_idx] = c
                print(f"Processed {z_idx+1} of {len(img_data)} image slices")        


            self.coords_array = coords_array
            self.coords_array_img = coords_array_img
            self.image_array = img_array

        elif img_type == "diffusion" and load_method == "dicom":

            b_value_img_dict, meta_attrs = get_diffusion_image_metadata(os.path.join(data_dir, "dicom_files"))
            #now save the data into image object with images sorted in a dictionary by b value, with 3d image fore each
            ipps = meta_attrs["ipps"]
            iop = meta_attrs["iop"]
            self.iop = iop
            self.name = str(meta_attrs["patient_name"])
            self.acquisition_date = meta_attrs["acquisition_date"]
            self.modality = meta_attrs["modality"]
            self.origin = meta_attrs["origin"]
            x_orig = float(self.origin[0])
            y_orig = float(self.origin[1])
            z_orig = float(self.origin[2])
        # we will refer to patient cartesian coordinates as x,y,z and the image coordinates as a,b,c
            self.cos_ax = float(iop[0])
            self.cos_ay = float(iop[1])
            self.cos_az = float(iop[2])
            self.cos_bx = float(iop[3])
            self.cos_by = float(iop[4])
            self.cos_bz = float(iop[5])

            #get cosines with c by taking cross product. 
            cross_prod = np.cross(np.array(iop[0:3]), np.array(iop[3:]))
            self.cos_cx = cross_prod[0]
            self.cos_cy = cross_prod[1]
            self.cos_cz = cross_prod[2]

            self.slice_thickness = meta_attrs["slice_thickness"]   #comes from space between slices attr
            self.pixel_spacing = meta_attrs["pixel_spacing"]
            # pixel_spacing_list = [metadata_list[i].PixelSpacing for i in range(100,len(metadata_list))]
            # slice_thickness_list = [np.linalg.norm(np.array(metadata_list[i+1].ImagePositionPatient) - np.array(metadata_list[i].ImagePositionPatient)) for i in range(100,len(metadata_list)-1)]
            img_list = []
            

            
            #sort b values
            b_value_img_dict_sorted = {}
            b_values = list(b_value_img_dict.keys())
            b_values.sort()
                #need to convert to suvbw 
            row_count, col_count = b_value_img_dict[b_values[0]][str(ipps[0])].shape
            #make 3d image for each b value
            for b in b_values:
                num_slices = len(b_value_img_dict[b])
                img_array = np.zeros([num_slices, row_count, col_count], dtype=np.float32) 
                for i in range(len((ipps))):
                    img_array[i,:,:] = b_value_img_dict[b][str(ipps[i])]
                b_value_img_dict_sorted[b] = img_array
            #get the coordinates array to know the cartesian coordinates in each voxel center.    
            coords_array = np.zeros([3,num_slices, row_count, col_count], dtype=np.float32)    #dicom geometry
            coords_array_img = np.zeros([3,num_slices, row_count, col_count], dtype=np.float32)   #converted to image geometry.
            
            for z_idx in range(len(ipps)):
                corner_x, corner_y, corner_z = ipps[z_idx]

                for y_idx in range(row_count):
                    b= y_idx * self.pixel_spacing[0]
                    for x_idx in range(col_count):
                        a = x_idx * self.pixel_spacing[1] 
                        #c = z_idx * self.slice_thickness
                        c = (corner_z - z_orig) * self.cos_cz
                        z = corner_z + (b * self.cos_bz) + (a * self.cos_az)
                        x = corner_x + (b * self.cos_bx) + (a * self.cos_ax)
                        y = corner_y + (b * self.cos_by) + (a * self.cos_ay)
                        # test_a = (x-x_orig) * self.cos_ax + (y-y_orig) * self.cos_ay+ (z-z_orig) * self.cos_az
                        # test_b = (x-x_orig) * self.cos_bx + (y-y_orig) * self.cos_by+ (z-z_orig) * self.cos_bz
                        # test_c = (x-x_orig) * self.cos_cx + (y-y_orig) * self.cos_cy+ (z-z_orig) * self.cos_cz
                        coords_array[0,z_idx,y_idx, x_idx] = x
                        coords_array[1,z_idx,y_idx, x_idx] = y
                        coords_array[2,z_idx,y_idx, x_idx] = z

                        coords_array_img[0,z_idx,y_idx, x_idx] = a
                        coords_array_img[1,z_idx,y_idx, x_idx] = b
                        coords_array_img[2,z_idx,y_idx, x_idx] = c
                print(f"Processed {z_idx+1} of {num_slices} image slices")        


            self.coords_array = coords_array
            self.coords_array_img = coords_array_img
            self.image_arrays = b_value_img_dict_sorted 
def HHMMSS_To_S(val):
    s = 0
    hours = float(val[0:2]) * 60 * 60
    mins = float(val[2:4]) * 60
    seconds = float(val[4:])
    s = hours + mins + seconds
    return s

def get_hume_lbm(weight, height, sex):
    if sex == 'M':
        return 0.3281*weight + 0.33929*height - 29.5336
    elif sex == 'F':
        return 0.29569 * weight + 0.41813 * height - 43.2933    
    else: 
        return 0.5 * (0.3281*weight + 0.33929*height - 29.5336         +        0.29569 * weight + 0.41813 * height - 43.2933)



def get_img_series_metadata(img_dir, modality="ct"):
    #will return a list which has the metadata for each image in slice order. 
    #list is sorted from smallest to largest z. 
    files_loaded = 0
    file_paths = glob.glob(os.path.join(img_dir, "*"))
    img_list = []
    for path in file_paths:
        try:
            metadata = pydicom.dcmread(path)
            files_loaded += 1
            print(f"Loaded {files_loaded} of {len(file_paths)} DICOM images.")
        except:
            print(f"Could not load {path}. Continuing...")  
            continue  
        if modality.lower() == "pet":
            if "pt"  not in metadata.Modality.lower() and "pet"  not in metadata.Modality.lower():
                continue
        elif modality.lower() != metadata.Modality.lower():
            continue
        img_list.append(metadata)
    img_list.sort(key=lambda x: x.ImagePositionPatient[2])
    return img_list

def get_img_series_array(img_dir):
    
    #    
    meta_list = get_img_series_metadata(img_dir)
    
def convert_contours_to_img_coords(img_series, structures_dict):
    structures_dict = copy.deepcopy(structures_dict)
    origin = img_series.origin
    x_orig = origin[0]
    y_orig = origin[1]
    z_orig = origin[2]
    cos_ax = img_series.cos_ax 
    cos_ay = img_series.cos_ay 
    cos_az = img_series.cos_az 
    cos_bx = img_series.cos_bx 
    cos_by = img_series.cos_by
    cos_bz = img_series.cos_bz
    cos_cx = img_series.cos_cx
    cos_cy = img_series.cos_cy
    cos_cz = img_series.cos_cz
    for structure_name in structures_dict:
        new_whole_roi = []
        contours = structures_dict[structure_name]
        whole_roi = contours.wholeROI
        for contour in whole_roi:
            for island in contour:
                new_contour = []
                for point in island:
                    #first get distance of x,y,z from image origin
                    x_rel = point[0] - x_orig
                    y_rel = point[1] - y_orig
                    z_rel = point[2] - z_orig

                    #now get converted points (x,y,z) --> (a,b,c)
                    a = (x_rel * cos_ax) + (y_rel * cos_ay) + (z_rel * cos_az)
                    b = (x_rel * cos_bx) + (y_rel * cos_by) + (z_rel * cos_bz)
                    c = (x_rel * cos_cx) + (y_rel * cos_cy) + (z_rel * cos_cz)

                    new_contour.append([a,b,c])
            new_whole_roi.append([new_contour])     
        contours.whole_roi_img = new_whole_roi           
    return structures_dict  

   

def get_contours_on_img_planes(img_series, structures_dict, segmentation_types=["reg", "si", "ap", "ml"], img_type="reg"):
    #this function will convert the contours lists such that all contours align with image planes in img_series. 
    #if img type == deblurred, will get contours on the deblurred x2 image planes. 
    #first get the img coord "c" value of each img.
    c_vals = []
    if img_type == "reg":
        coords_array_img = img_series.coords_array_img
    elif img_type == "deblur":
        coords_array_img = img_series.deconv_coords_2x_img
    slice_thickness = float(img_series.slice_thickness)
    for c in coords_array_img[2,:,0,0]:
        c_vals.append(c)
    c_vals = np.array(c_vals)


    for structure_name in structures_dict:
        new_contours_whole = []    
        contours_whole = structures_dict[structure_name].whole_roi_img
        for contour in contours_whole:
            if len(contour) == 0:
                continue
            for island in contour:
                c = island[0][2]
                closest_indices = np.argsort(np.abs(c_vals-c))
                closest_img_c = c_vals[closest_indices[0]]     #find two closest image slices c vals


                if np.abs(closest_img_c-c) > slice_thickness:
                    print(f"Couldn't find image slice in range for contour at c = {c}")
                    continue
                elif np.abs(closest_img_c-c) <= 0.5:
                    new_contour = []
                    for point in island:
                        new_point = [point[0], point[1], closest_img_c] 
                        new_contour.append(new_point)
                    new_contours_whole.append([new_contour])    
                elif np.abs(closest_img_c-c) < slice_thickness:
                    print(f"Warning: contour for {structure_name} at c = {c} assigned to image slice more than 0.5mm away, but less than slice thickness.")
                    new_contour = []
                    for point in island:
                        new_point = [point[0], point[1], closest_img_c] 
                        new_contour.append(new_point)
                    new_contours_whole.append([new_contour])  
        if img_type == "reg":            
            structures_dict[structure_name].whole_roi_img_planes = new_contours_whole    
        elif img_type == "deblur":
            structures_dict[structure_name].whole_roi_img_planes_deblur = new_contours_whole            

        #now get the subsegment contours on the image planes     
        for segmentation_type in segmentation_types:       
            new_subsegmented_contours = []
            if not hasattr(structures_dict[structure_name], str("segmented_contours_" + segmentation_type)):
                continue
            subsegmented_contours = getattr(structures_dict[structure_name], str("segmented_contours_" + segmentation_type))

            if subsegmented_contours == None:
                continue

            for subsegment in subsegmented_contours:
                new_subsegment_contour = []
                for contour in subsegment:
                    if len(contour) == 0:
                        continue
                    for island in contour:
                        if len(island) == 0:
                            continue
                        c = island[0][2]
                        closest_indices = np.argsort(np.abs(c_vals-c))
                        closest_img_c = c_vals[closest_indices[0]]     #find two closest image slices c vals

                        if np.abs(closest_img_c-c) > slice_thickness:
                            print(f"Couldn't find image slice in range for contour at c = {c}")
                            continue
                        elif np.abs(closest_img_c-c) <= 0.5:
                            new_contour = []
                            for point in island:
                                new_point = [point[0], point[1], closest_img_c] 
                                new_contour.append(new_point)
                            new_subsegment_contour.append([new_contour])    
                        elif np.abs(closest_img_c-c) < slice_thickness:
                            print(f"Warning: contour for {structure_name} at c = {c} assigned to image slice more than 0.5mm away, but less than slice thickness.")
                            new_contour = []
                            for point in island:
                                new_point = [point[0], point[1], closest_img_c] 
                                new_contour.append(new_point)
                            new_subsegment_contour.append([new_contour])      
                new_subsegmented_contours.append(new_subsegment_contour)     
            if img_type == "reg":    
                setattr(structures_dict[structure_name], str("segmented_contours_" + segmentation_type + "_img_planes"), new_subsegmented_contours)       
            elif img_type == "deblur":    
                setattr(structures_dict[structure_name], str("segmented_contours_" + segmentation_type + "_img_planes_deblur"), new_subsegmented_contours)                   
            #structures_dict[structure_name].segmented_contours_img_planes = new_subsegmented_contours
    return structures_dict



def save_all_img_and_mask_as_nrrd(img_series, structures_dict, save_paths=os.getcwd(), clear_existing=True, suv_factors=None, segmentation_types=["reg"], deblurred=False):
    if clear_existing==True:    #clear existing files in nrrd directories
        if type(save_paths) is list:    
            for r in range(2):
                for file in os.listdir(save_paths[r]):
                    os.remove(os.path.join(save_paths[r], file))
        elif type(save_paths) == str:
            for file in os.listdir(save_paths):
                os.remove(os.path.join(save_paths, file))
             
    for structure_name in structures_dict:
        save_img_and_mask_as_nrrd(img_series, structure_name, structures_dict[structure_name], save_paths, suv_factors=suv_factors, segmentation_types=["reg"], deblurred=deblurred)
        print(f"Current time: {datetime.datetime.now().time()}")
    return
def zero_borders(array):
    #this makes all border values in a 3d array = 0
    array[0, :, :] = 0
    array[-1, :, :] = 0
    array[:, 0, :] = 0
    array[:, -1, :] = 0
    array[:, :, 0] = 0
    array[:, :, -1] = 0
    return array

def crop_around_mask(array):
    non_zeros = np.nonzero(array)
    min_z = min(non_zeros[0]) -5
    max_z = max(non_zeros[0]) + 5  

    min_y = min(non_zeros[1]) - 5
    max_y = max(non_zeros[1]) + 5

    min_x = min(non_zeros[2]) - 5
    max_x = max(non_zeros[2]) + 5

    new_array = array[min_z:max_z, min_y:max_y, min_x:max_x]
    boundaries = [min_z, min_y, min_x, max_z, max_y, max_x]
    return new_array, boundaries

            
def centre_roi_mask_img(img):
    vals = np.where(img != 0)
    #get the centre coordinates 
    centres = list(img.shape) 
    for i in range(len(centres)):
        centres[i] /= 2
    avg_0 = int(np.average(vals[0]) - centres[0])
    avg_1 = int(np.average(vals[1]) - centres[1])
    avg_2 = int(np.average(vals[2]) - centres[2])


    
    #now subtract average from each pixel that is a 1. 
    new_img = np.zeros(img.shape)
    shape = img.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                new_i = i - avg_0
                new_j = j - avg_1
                new_k = k - avg_2

                if new_i >= 0 and new_i < shape[0] and new_j >= 0 and new_j < shape[1] and new_k >= 0 and new_k < shape[2]:
                    new_img[new_i, new_j, new_k] = img[i,j,k] 
    
    # vals = np.where(new_img != 0)
    # #get the centre coordinates 
    # centres = list(new_img.shape) 
    # for i in range(len(centres)):
    #     centres[i] /= 2
    # avg_0 = int(np.average(vals[0]) - centres[0])
    # avg_1 = int(np.average(vals[1]) - centres[1])
    # avg_2 = int(np.average(vals[2]) - centres[2])
    return new_img


def save_img_and_mask_as_nrrd(img_series, structure_name, structure, save_paths, suv_factors, segmentation_types=["reg"], deblurred=False):
    if suv_factors==None:
        img_file_name = img_series.name + "_" + img_series.acquisition_date + ".nrrd"
        struct_file_name = img_series.name + "_" + img_series.acquisition_date + "__" + structure_name + "__"  #put whole or ss number after, and .nrrd
        if deblurred == False or img_series.modality == "CT":
            whole_roi_mask = structure.whole_roi_masks
        elif deblurred == True:
            whole_roi_mask = structure.whole_roi_masks_deblur
        if type(save_paths) is list and len(save_paths) > 1:
            img_save_path = os.path.join(save_paths[0], img_file_name)
            struct_save_path = os.path.join(save_paths[1], struct_file_name)
        elif type(save_paths) is list and len(save_paths) == 1:
            img_save_path = os.path.join(save_paths[0], img_file_name)
            struct_save_path = os.path.join(save_paths[0], struct_file_name)
        elif type(save_paths) == str:
            img_save_path = os.path.join(save_paths, img_file_name)
            struct_save_path = os.path.join(save_paths, struct_file_name)    
        if save_paths is None:  
            img_save_path = os.path.join(os.getcwd(), img_file_name) 
            struct_save_path = os.path.join(os.getcwd(), struct_file_name)   
        
        # from utils import plot_3d_image
        # plot_3d_image(img_series.image_array)
        # plot_3d_image(whole_roi_mask) 
        #first need to swap the rows and columns of image and masks because nrrd wants first dimension to be width, second to be height . also sort from largest z to smallest instead of smallest to largest
        img = np.swapaxes(img_series.image_array, 0,2)
        
        #make the header
        header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['mm','mm', 'mm'], 'spacings': [float(img_series.pixel_spacing[1]), float(img_series.pixel_spacing[0]), float(img_series.slice_thickness)]} #'space directions': np.array([[1,0,0], [0,1,0],[0,0,1]])

        nrrd.write(img_save_path, img, header)    
        print(f"Wrote nrrd file to {img_save_path}")


        whole_roi_mask = np.swapaxes(whole_roi_mask,0,2).astype(int)    

        nrrd.write(str(struct_save_path + "whole__.nrrd"), whole_roi_mask, header)
        print(f"Wrote nrrd file data to {str(struct_save_path + 'whole__.nrrd')}")
        for seg_type in segmentation_types:
            if deblurred == False or img_series.modality == "CT":
                subseg_masks = getattr(structure, f"subseg_masks_{seg_type}")
            else:
                subseg_masks = getattr(structure, f"subseg_masks_{seg_type}_deblur")    
            for s,subseg in enumerate(subseg_masks):
                subseg = np.swapaxes(copy.deepcopy(subseg), 0, 2).astype(int)

                nrrd.write(str(struct_save_path + str(s) + "__.nrrd"), subseg, header)
                print(f"Wrote nrrd file data to {struct_save_path + str(s) + '__.nrrd'}")

        return
    else:
        suv_factor_names = ["bw", "lbm", "bsa"] 
        for s, suv_factor in enumerate(suv_factors):
            img_file_name = img_series.name + "_" + img_series.acquisition_date + "__" + suv_factor_names[s] + "__.nrrd"
            struct_file_name = img_series.name + "_" + img_series.acquisition_date + "__" + structure_name + "__"  #put whole or ss number after, and .nrrd
            if deblurred == False or img_series.modality == "CT":
                whole_roi_mask = structure.whole_roi_masks
            elif deblurred == True:
                whole_roi_mask = structure.whole_roi_masks_deblur

            if type(save_paths) is list and len(save_paths) > 1:
                img_save_path = os.path.join(save_paths[0], img_file_name)
                struct_save_path = os.path.join(save_paths[1], struct_file_name)
            elif type(save_paths) is list and len(save_paths) == 1:
                img_save_path = os.path.join(save_paths[0], img_file_name)
                struct_save_path = os.path.join(save_paths[0], struct_file_name)
            elif type(save_paths) == str:
                img_save_path = os.path.join(save_paths, img_file_name)
                struct_save_path = os.path.join(save_paths, struct_file_name)    
            if save_paths is None:  
                img_save_path = os.path.join(os.getcwd(), img_file_name) 
                struct_save_path = os.path.join(os.getcwd(), struct_file_name)   
            
            # from utils import plot_3d_image
            # plot_3d_image(img_series.deconv_array / img_series.suv_factors[1] * suv_factor)
            # plot_3d_image(whole_roi_mask) 
            #first need to swap the rows and columns of image and masks because nrrd wants first dimension to be width, second to be height . also sort from largest z to smallest instead of smallest to largest
            if deblurred == False or img_series.modality == "CT":
                img = np.swapaxes(img_series.image_array, 0,2) * suv_factor
            else:
                img = img_series.deconv_array / img_series.suv_factors[1] * suv_factor
                img = np.swapaxes(img, 0,2)
            #make the header
            header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['mm','mm', 'mm'], 'spacings': [float(img_series.pixel_spacing[1]), float(img_series.pixel_spacing[0]), float(img_series.slice_thickness)]} #'space directions': np.array([[1,0,0], [0,1,0],[0,0,1]])

            nrrd.write(img_save_path, img, header)    
            print(f"Wrote nrrd file to {img_save_path}")

            if s == 0: #only do once
                whole_roi_mask = np.swapaxes(whole_roi_mask,0,2).astype(int)     

                nrrd.write(str(struct_save_path + "whole__.nrrd"), whole_roi_mask, header)
                print(f"Wrote nrrd file data to {str(struct_save_path + 'whole__.nrrd')}")

                for seg_type in segmentation_types:
                    if deblurred == False or img_series.modality == "CT":
                        subseg_masks = getattr(structure, f"subseg_masks_{seg_type}")
                    else:
                        subseg_masks = getattr(structure, f"subseg_masks_{seg_type}_deblur")   
                    for s_idx,subseg in enumerate(subseg_masks):
                        subseg = np.swapaxes(copy.deepcopy(subseg), 0, 2).astype(int)

                        nrrd.write(str(struct_save_path + str(s_idx) + "__.nrrd"), subseg, header)
                        print(f"Wrote nrrd file data to {struct_save_path + str(s_idx) + '__.nrrd'}")


def convert_all_dicoms_to_nrrd(modality, save_paths=None, patient_num=None, deblurred=False):   
    #before running, must process all image and mask arrays with the desired subsegmentation scheme
    print("Beginning conversion of all dicoms to nrrd images.")
    print(f"Current time: {datetime.datetime.now().time()}")

    if deblurred==False:
        img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")
    elif deblurred==True:
        img_series_path = os.path.join(os.getcwd(), "data_deblur", patient_num, "img_dict")   
    with open(img_series_path, "rb") as fp:
        img_dict = pickle.load(fp)
    #load masks 
    with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
        mask_dict = pickle.load(fp)

    structures_masks_dict= mask_dict[modality]

    img_series = img_dict[modality]
    if modality == "PET":
        suv_factors = img_dict["PET"].suv_factors
    else:
        suv_factors = None    

    del_structures = []

    for structure in structures_masks_dict:
        if "par" in structure.lower() and "sup" not in structure.lower():
            #Chopper.organ_chopper(structures_masks_dict[structure], subsegmentation)
            continue 
        else:
            del_structures.append(structure)
    for val in del_structures:
        del structures_masks_dict[val]            
    print(f"Current time: {datetime.datetime.now().time()}")

 
    save_all_img_and_mask_as_nrrd(img_series, structures_masks_dict, save_paths=save_paths, clear_existing=True, suv_factors=suv_factors, segmentation_types=["reg"], deblurred=deblurred)

def is_substring_present(string, str_list):
    for s in str_list:
        if s in string:
            return True
    return False        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="converter for dicom images and structures to make corresponding nrrd files for use with pyradiomics.")
    parser.add_argument("--img_dir", help="specify the folder containing a single patient's image series.", default=None, type=str)
    parser.add_argument("--structure_dir", help="specify the folder containing a single patient's RTSTRUCT file.", default=None, type=str)
    parser.add_argument("--modality", help="specify image modality you wish to convert (Default is CT. Other options are PET or MR)", default="None", type=str)
    parser.add_argument("--save_dir", help="Specify where you wish to save your nrrd files. if You wish to place converted structures in a separate folder, also specify the argument \"save_dir_struct\"", default=None, type=str)
    parser.add_argument("--save_dir_struct", help="Specify a separate folder you'd like to save the struct nrrd files to.", default=None, type=str)
    args = parser.parse_args()
    v = vars(args)
    n_args = sum([1 for a in v.values() if a])
    print("Starting DICOM to NRRD converter.")
    print(f"Supplied image directory: {args.img_dir}")
    print(f"Supplied structure directory: {args.structure_dir}")
    print(f"Supplied output directory: {args.save_dir}")
    print(f"Supplied modality: {args.modality}")

    img_dir = None
    structure_dir = None
    modality = "CT"
    save_dir = None
    if os.path.exists(args.img_dir):
        img_dir = args.img_dir
    if os.path.exists(args.structure_dir):
        structure_dir = args.structure_dir
    if os.path.exists(args.save_dir):
        save_dir = args.save_dir
    if args.img_dir == None:
        while True:
            try:
                input_val = input("\n Please specify the directory containing dicom data for a single patient image series. If structure(s) are in a different folder, please separate directories with a comma.\n>> ")
                input_val = list(input_val.split(","))

                if len(input_val) == 1:    #same directory for images/structures
                    if os.path.exists(input_val[0].strip()):
                        print(f"Loading all data from {input_val}")
                        img_dir =input_val[0].strip()
                        structure_dir = input_val[0].strip()
                        break
                if len(input_val) == 2:    #separate dir for images/structures
                    input_val[0] = input_val[0].strip()    #remove any white space in path
                    input_val[1] = input_val[1].strip()    #remove any white space in path
                    if os.path.exists(input_val[0]) and os.path.exists(input_val[1]):
                        print(f"Loading image data from {input_val[0]}")
                        print(f"Loading structure data from {input_val[1]}")
                        img_dir = input_val[0]
                        structure_dir = input_val[1]
                        break
                if len(input_val) >= 3:
                   print("More than two paths specified. Please include up to 2 separate paths (one for images, one for structures)") 

            except KeyboardInterrupt:
                quit()
            except: pass    
    elif not os.path.exists(args.img_dir):
        while True:
            try:
                input_val = input(f"\n Could not find {args.img_dir}. Please specify a new directory containing image data. If structure(s) are in a different folder, please separate directories with a comma.\n>> ")
                input_val = list(input_val.split(","))

                if len(input_val) == 1:    #same directory for images/structures
                    if os.path.exists(input_val[0].strip()):
                        print(f"Loading all data from {input_val}")
                        img_dir =input_val[0].strip()
                        structure_dir = input_val[0].strip()
                        break
                if len(input_val) == 2:    #separate dir for images/structures
                    input_val[0] = input_val[0].strip()    #remove any white space in path
                    input_val[1] = input_val[1].strip()    #remove any white space in path
                    if os.path.exists(input_val[0]) and os.path.exists(input_val[1]):
                        print(f"Loading image data from {input_val[0]}")
                        print(f"Loading structure data from {input_val[1]}")
                        img_dir = input_val[0]
                        structure_dir = input_val[1]
                        break
                if len(input_val) >= 3 or len(input_val) == 0:
                   print("0 or more than two paths specified. Please include up to 2 separate paths (one for images, one for structures)\n>> ") 

            except KeyboardInterrupt:
                quit()
            except: pass  

    if args.structure_dir == None:
        structure_dir = img_dir
        
    if args.save_dir == None:
        while True:
            try:
                input_val = input("\n Please specify the directory to save nrrd images to. If structure(s) are to be saved in a different folder, please separate directories with a comma.\n>> ")
                input_val = list(input_val.split(","))
                if len(input_val) == 1:
                    if os.path.exists(input_val[0].strip()):
                        save_dir = input_val[0].strip()
                        print(f"saving all nrrd data into {input_val}")
                        break
                if len(input_val) == 2:
                    input_val[0] = input_val[0].strip()    #remove any white space in path
                    input_val[1] = input_val[1].strip()    #remove any white space in path
                    if os.path.exists(input_val[0]) and os.path.exists(input_val[1]):
                        print(f"saving image nrrd data into {input_val[0]}")
                        print(f"saving structure nrrd data into {input_val[1]}")
                        save_dir = input_val
                        break
                if len(input_val) >= 3:
                   print("More than two paths specified. Please include up to 2 separate paths (one for images, one for structures)\n>> ") 

            except KeyboardInterrupt:
                quit()
            except: pass   
    elif not os.path.exists(args.save_dir):
        while True:
            try:
                input_val = input(f"\n Could not find {args.save_dir}. Please specify the directory to save nrrd images to. If structure(s) are to be saved in a different folder, please separate directories with a comma.\n>> ")
                input_val = list(input_val.split(","))
                if len(input_val) == 1:
                    if os.path.exists(input_val[0].strip()):
                        save_dir = input_val[0].strip()
                        print(f"saving all nrrd data into {input_val}")
                        break
                if len(input_val) == 2:
                    input_val[0] = input_val[0].strip()    #remove any white space in path
                    input_val[1] = input_val[1].strip()    #remove any white space in path
                    if os.path.exists(input_val[0]) and os.path.exists(input_val[1]):
                        print(f"saving image nrrd data into {input_val[0]}")
                        print(f"saving structure nrrd data into {input_val[1]}")
                        save_dir = input_val
                        break
                if len(input_val) >= 3:
                   print("More than two paths specified. Please include up to 2 separate paths (one for images, one for structures)\n>> ") 

            except KeyboardInterrupt:
                quit()
            except: pass       

    if args.save_dir_struct is not None and not os.path.exists(args.save_dir_struct):
        while True:
            try:
                input_val = input(f"\n Could not find {args.save_dir_struct} for structure nrrd saving. Please re-specify the directory to save nrrd images to for structures. If structure(s) are to be saved in a different folder, please separate directories with a comma.\n>> ")
                input_val = list(input_val.split(","))
                if len(input_val) == 1:
                    if os.path.exists(input_val[0].strip()):
                        save_dir = input_val[0].strip()
                        print(f"saving all nrrd data into {input_val}")
                        break
                if len(input_val) == 2:
                    input_val[0] = input_val[0].strip()    #remove any white space in path
                    input_val[1] = input_val[1].strip()    #remove any white space in path
                    if os.path.exists(input_val[0]) and os.path.exists(input_val[1]):
                        print(f"saving image nrrd data into {input_val[0]}")
                        print(f"saving structure nrrd data into {input_val[1]}")
                        save_dir = input_val
                        break
                if len(input_val) >= 3:
                   print("More than two paths specified. Please include up to 2 separate paths (one for images, one for structures)") 

            except KeyboardInterrupt:
                quit()
            except: pass      
    elif args.save_dir_struct is not None and os.path.exists(args.save_dir_struct) and len(save_dir) == 1:
        save_dir = [save_dir[0], args.save_dir_struct]          
    if args.modality == None:
        print("no modality specified, using CT as default.")
    
    convert_all_dicoms_to_nrrd(img_dir, structure_dir, modality, save_dir)    
