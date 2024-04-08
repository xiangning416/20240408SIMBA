import numpy as np
from skimage.measure import find_contours
from skimage.transform import rescale
from shapely.geometry import Polygon
import numpy as np
from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from typing import List, Dict, Tuple
import zarr
import matplotlib.pyplot as plt
import uuid
import datetime
import os

class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)
    
class KolabSegment(analysistask.AnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'starting_z' not in self.parameters:
            self.parameters['starting_z'] = 2

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [
                self.parameters['global_align_task']]
    
    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)
    
    def _run_analysis(self) -> None:
        path_to_zarrs = self.parameters['path_to_segmentation_zarrs']
        zarr_name = self.parameters['zarr_name']
        scale_factor = self.parameters['scale_factor']
        imageSize = self.dataSet.get_image_dimensions()
        microns_per_pixel = 0.107 #self.dataSet.get_microns_per_pixel()

        # mask_matrix = rescale(zarr.open(os.path.join(path_to_zarrs, zarr_name))[:], [1, 2, 2], order=0) #upscale s1 -> s0
        mask_matrix = zarr.open(os.path.join(path_to_zarrs, zarr_name))[:]

        positions = self.dataSet.get_stage_positions()
        minx = min(positions.X) #only works if x has negative positions -- TODO change to subtracting absolute value of min 
        miny = min(positions.Y)

        allz_boundaries = []
        micron_allz_boundaries = []

        for z_level in range(len(mask_matrix)):
            pixel_boundaries = find_contours(np.transpose(mask_matrix[z_level]), 0.9, fully_connected='high') #extract boundaries for each z level 
            micron_boundaries = [np.asarray([[((coord[0]*scale_factor)*microns_per_pixel + minx), (coord[1]*scale_factor)*microns_per_pixel + miny] for coord in bound]) for bound in pixel_boundaries] #shift boundaries to match stage positions
            micron_allz_boundaries.append(micron_boundaries) #append boundaries for each z level to list
            allz_boundaries.append(pixel_boundaries) #append boundaries for each z level to list
        
        valid_boundaries = []
        micron_valid_boundaries = []
        for z in range(len(mask_matrix)):
            valid_z_boundaries = [] #stores all valid cells for each z level
            micron_valid_z_boundaries = []
            for i in range(len(allz_boundaries[z])): #loop through each cell in each z level
                if(len(allz_boundaries[z][i]) > 3):
                    if Polygon(allz_boundaries[z][i]).is_valid and Polygon(allz_boundaries[z][i]).area > 1: #remove lines & invalid polygons
                        valid_z_boundaries.append(allz_boundaries[z][i])
                        micron_valid_z_boundaries.append(micron_allz_boundaries[z][i])
                    # else: 
                    #     print("invalid cell")
            valid_boundaries.append(valid_z_boundaries)
            micron_valid_boundaries.append(micron_valid_z_boundaries)
        allz_boundaries = valid_boundaries
        micron_allz_boundaries = micron_valid_boundaries 

        bound_dict = {}
        for z in range(len(allz_boundaries)):
            contour_num = 0
            for bound in allz_boundaries[z]:

                cellid = 0
                poly_coords = Polygon(bound).exterior.coords
                
                for coord in poly_coords:
                    y, x = int(coord[1]), int(coord[0])
                    if(mask_matrix[z][y][x] != 0):
                        cellid = mask_matrix[z][y][x]
                        break
                #TODO check this!
                try:
                    #if cell already exists in dictionary
                    bound_dict[cellid][(z+self.parameters['starting_z'])*1.5].append(Polygon(micron_allz_boundaries[z][contour_num])) #add new geometry to list (allows for multiple geometries per cell) -- in micron coordinates
                except:
                    try:
                        bound_dict[cellid][(z+self.parameters['starting_z'])*1.5] = [Polygon(micron_allz_boundaries[z][contour_num])] # use list to allow for multiple geometries per cell
                    except:
                        bound_dict[cellid] = {} # add z & use list to allow for multiple geometries per cell
                        bound_dict[cellid][(z+self.parameters['starting_z'])*1.5] = [Polygon(micron_allz_boundaries[z][contour_num])]
                contour_num +=1
        featureDB = self.get_feature_database() #get feature database -- this will store all the information about each cell  
        
        featurelist = []
        for cell_id in bound_dict.keys():
            cellpolygons = []
            z_levels = []
            for z in bound_dict[cell_id].keys():
                cellpolygons.append(bound_dict[cell_id][z])
                z_levels.append(z)
            
            featurelist.append(spatialfeature.SpatialFeature(cellpolygons, fov=0, zCoordinates = z_levels, uniqueID=cell_id))       
        featureDB.write_features(featurelist, fov=0) #write features to feature database for this fov
                
class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName)