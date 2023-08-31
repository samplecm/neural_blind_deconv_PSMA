
class Contours(object):

    
    def __init__(self, name, dicomName, contours):
        self.wholeROI = contours
        self.segmentedContours = None
        self.roiName = name
        self.dicomName = dicomName
        self.dose_params = None
        self.dose_bins = None
        self.dose_voxels = None 
        self.dose_voxels_subsegs = None
        self.centre_point = None 
        self.centre_point_subsegs = None
        self.spatial_relationships = None 