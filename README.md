# neural_blind_deconv_PSMA
This code corresponds to the 2023 work: Neural blind deconvolution for simultaneous partial volume effect correction and super-sampling of PSMA PET images by Caleb Sample, Arman Rahmim, Carlos Uribe, Francois Benard, Jonn Wu, and Haley Clark. 
(https://iopscience.iop.org/article/10.1088/1361-6560/ad36a9)


Here, neural blind deconvolution, as first roposed by Ren et al. 2020 (http://doi.org/10.1109/CVPR42600.2020.00340.) is adapted for pve mitigation/super-sampling of PSMA PET images.

See blind_deconv.py for the main training function. The models used are in model_deconv.py and the loss functions in loss_functions.py.

Note that prior to deblurring, DICOM images were first processed into Img_Series objects (image_processing.py) and masks were processed according to image anatomy from contours objects (contours.py).
