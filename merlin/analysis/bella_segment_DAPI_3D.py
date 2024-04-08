import numpy as np
from skimage.measure import find_contours
from skimage.transform import rescale
from shapely.geometry import Polygon
from shapely.strtree import STRtree
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
    
class KolabSegmentDAPI3D(analysistask.AnalysisTask):

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
        if 'starting_z' not in self.parameters:
                    self.parameters['starting_z'] = 2

        starting_z = self.parameters['starting_z']
        # mask_matrix = rescale(zarr.open(os.path.join(path_to_zarrs, zarr_name))[:], [1, 2, 2], order=0) #upscale s1 -> s0
        eroded = zarr.open(os.path.join(path_to_zarrs, zarr_name))[:]

        positions = self.dataSet.get_stage_positions()
        minx = min(positions.X) #only works if x has negative positions -- TODO change to subtracting absolute value of min 
        miny = min(positions.Y)

        factor = 0
        for z in range(len(eroded)):
            eroded[z] = np.where(eroded[z] != 0, eroded[z] + factor, eroded[z])
            factor += len(np.unique(eroded[z]))

        zPolygons = [[] for _ in range(len(eroded))]
        zIDs = [[] for _ in range(len(eroded))]
        nuclear_polys = [[] for _ in range(len(eroded))]

        for z in range(len(eroded) - 1): #len(touch_eroded) - 1

            if z == 0:
                touch_eroded_arr = eroded[z]
                pixel_boundaries = find_contours(np.transpose(touch_eroded_arr), 0.9, fully_connected='high')
                nuclei_contours = [np.asarray([[((coord[0]*scale_factor)*microns_per_pixel + minx), (coord[1]*scale_factor)*microns_per_pixel + miny] for coord in bound]) for bound in pixel_boundaries] #shift boundaries to match micron stage positions
                
                for i, contour in enumerate(pixel_boundaries): 
                    #find corresponding mask for each contour
                    for coord in contour:
                        y, x = int(coord[1]), int(coord[0])
                        if(touch_eroded_arr[y][x] != 0):
                            nuclearID = touch_eroded_arr[y][x]
                            if z == 0:
                                zPolygons[z].append(Polygon(nuclei_contours[i]))
                                zIDs[z].append(nuclearID)
                            break
                            
            z1_touch_eroded_arr = eroded[z+1]
            pixel_boundaries = find_contours(np.transpose(z1_touch_eroded_arr), 0.9, fully_connected='high')
            nuclei_contours = [np.asarray([[((coord[0]*scale_factor)*microns_per_pixel + minx), (coord[1]*scale_factor)*microns_per_pixel + miny] for coord in bound]) for bound in pixel_boundaries] #shift boundaries to match micron stage positions
            
            for i, contour in enumerate(pixel_boundaries): 
                #find corresponding mask for each contour
                for coord in contour:
                    y, x = int(coord[1]), int(coord[0])
                    if(z1_touch_eroded_arr[y][x] != 0):
                        nuclearID = z1_touch_eroded_arr[y][x]
                        nuclear_polys[z+1].append({'polygon': Polygon(nuclei_contours[i]), 'nuclearID': nuclearID})
                        zPolygons[z+1].append(Polygon(nuclei_contours[i]))
                        zIDs[z+1].append(nuclearID)
                        break
            
            #create STRtree for Z above 
            tree = STRtree([nuclei["polygon"] for nuclei in nuclear_polys[z+1]])
            nuclearIDs = np.array([nuclei["nuclearID"] for nuclei in nuclear_polys[z+1]])
            print(len(zPolygons[z]))
            for id, nucleus in enumerate(zPolygons[z]):
                potential_nuclei = (nuclearIDs.take(tree.query_nearest(nucleus)).tolist())
                for pnuclei in potential_nuclei:
                    try: #if nucleus has already been assigned to 3D nuclei, do not assign it again 
                        if((zPolygons[z+1][zIDs[z+1].index(pnuclei)].intersection(nucleus).area / zPolygons[z+1][zIDs[z+1].index(pnuclei)].area >= 0.5)
                        | (nucleus.intersection(zPolygons[z+1][zIDs[z+1].index(pnuclei)]).area / nucleus.area >= 0.5)):
                            eroded[eroded == pnuclei] = zIDs[z][id]
                            indices = np.where(np.array(zIDs[z+1]) == pnuclei)[0]
                            for index in indices: 
                                zIDs[z+1][index] = zIDs[z][id]
                    except: pass 
        
        print((os.path.join(path_to_zarrs, zarr_name, "3D")))
        zarr.save(os.path.join(path_to_zarrs, f"{zarr_name}/3D"), eroded)

        ID_polygon_dict = {}
        for z in range(len(zIDs)):
            for i in range(len(zIDs[z])):
                try:
                    ID_polygon_dict[zIDs[z][i]][(starting_z+z)*1.5].append(zPolygons[z][i])
                except:
                    try:
                        ID_polygon_dict[zIDs[z][i]][(starting_z+z)*1.5] = []
                        ID_polygon_dict[zIDs[z][i]][(starting_z+z)*1.5].append(zPolygons[z][i])
                    except:
                        ID_polygon_dict[zIDs[z][i]] = {}
                        ID_polygon_dict[zIDs[z][i]][(starting_z+z)*1.5] = []
                        ID_polygon_dict[zIDs[z][i]][(starting_z+z)*1.5].append(zPolygons[z][i])

        featureDB = self.get_feature_database() #get feature database -- this will store all the information about each cell  
        
        featurelist = []
        for nuclear_ID in ID_polygon_dict.keys():
            nucleipolygons = []
            z_levels = []
            for z in ID_polygon_dict[nuclear_ID].keys():
                nucleipolygons.append(ID_polygon_dict[nuclear_ID][z])
                z_levels.append(z)
            # print(z_levels)
            featurelist.append(spatialfeature.SpatialFeature(nucleipolygons, fov=0, zCoordinates = z_levels, uniqueID=nuclear_ID))       
        featureDB.write_features(featurelist, fov=0) #write features to feature database for this fov
                
class ExportCellMetadataDAPI(analysistask.AnalysisTask):
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
        return []#self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName)