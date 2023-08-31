# neural_blind_deconv_PSMA
code corresponding to the 2023 work: Neural blind deconvolution for simultaneous partial volume effect correction and super-sampling of PSMA PET images.  
Here, neural blind deconvolution, as proposed by Ren et al. 2020 (http://doi.org/10.1109/CVPR42600.2020.00340.) is adapted for pve mitigation/super-sampling of PSMA PET images.

See blind_deconv.py for the main training function. The models used are in model_deconv.py and the loss functions in loss_functions.py.

Note that prior to deblurring, DICOM images were first processed into Img_Series objects (image_processing.py) and masks were processed according to image anatomy from contours objects (contours.py).
