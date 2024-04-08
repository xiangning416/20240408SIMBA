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
from rtree import index as rindex
from shapely import geometry

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
    
class KolabSegmentDAPI(analysistask.AnalysisTask):

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
                self.parameters['global_align_task'], 
                self.parameters['segmentation_task']]
    
    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)
    
    def _run_analysis(self) -> None:
        segmentationTask = self.dataSet.load_analysis_task(
            self.parameters['segmentation_task'])
        
        DAPI_npy = self.parameters['DAPI_npy']
        imageSize = self.dataSet.get_image_dimensions()
        microns_per_pixel = 0.107 #self.dataSet.get_microns_per_pixel()
        scale_factor = self.parameters['scale_factor']
        if 'starting_z' not in self.parameters:
            self.parameters['starting_z'] = 2

        positions = self.dataSet.get_stage_positions()
        minx = min(positions.X) #only works if x has negative positions -- TODO change to subtracting absolute value of min 
        miny = min(positions.Y)
        
        DAPI_segmentation = np.load(DAPI_npy)
        micron_allz_DAPI_boundaries = []
        print("segmentation loaded")
        #make all masks unique IDs across Zs
        counter = 0
        for z_level in range(len(DAPI_segmentation)): 
            #replace all values in matrix with unique values
            DAPI_segmentation[z_level][DAPI_segmentation[z_level] != 0] += counter
            counter += (max(np.unique(DAPI_segmentation[z_level]))) 

        print("find contours")
        nuclei_ids = [[] for _ in range(len(DAPI_segmentation))]
        #extract nuclei boundaries 
        for z_level in range(len(DAPI_segmentation)):
            DAPI_pixel_boundaries = find_contours(np.transpose(DAPI_segmentation[z_level]), 0.9, fully_connected='high') #extract boundaries for each z level 
            for nuclei_bound in DAPI_pixel_boundaries:
                poly_coords = Polygon(nuclei_bound).exterior.coords
                for coord in poly_coords:
                    y, x = int(coord[1]), int(coord[0])
                    if(DAPI_segmentation[z_level][y][x] != 0):
                        nuclei_id = DAPI_segmentation[z_level][y][x]
                        nuclei_ids[z_level].append(nuclei_id)
                        break
            micron_DAPI_boundaries = [np.asarray([[((coord[0]*scale_factor)*microns_per_pixel + minx), (coord[1]*scale_factor)*microns_per_pixel + miny] for coord in bound]) for bound in DAPI_pixel_boundaries]
            micron_allz_DAPI_boundaries.append(micron_DAPI_boundaries)

        #2D to 3D 
        print("2D -> 3D")
        #somehow keep track if a cell has already been assigned to cell (on this Z)
        #create below rtree
        merged_nuclei = []
        for z_level in range(0, len(DAPI_segmentation) -1):
            print("curr z", z_level)
            assignment_dict = {}
            cell_num = 0 
            above_rtree = rindex.Index()
            for ii, cell in enumerate(micron_allz_DAPI_boundaries[z_level+1]):
                above_rtree.insert(ii, Polygon(cell).bounds)
            print(len(micron_allz_DAPI_boundaries[z_level]))
            for counter, cell in enumerate(micron_allz_DAPI_boundaries[z_level]):
                assignment_dict[nuclei_ids[z_level][counter]] = False
                cellpoly = Polygon(cell)
                possible_cells = above_rtree.intersection(cellpoly.bounds)
                # print(cellpoly.area)
                # print(counter)
                # print(len(list(possible_cells)))
                # break
                for pcell in np.unique(list((possible_cells))):
                    pcellpoly = Polygon(micron_allz_DAPI_boundaries[z_level+1][pcell])
                    cell_overlap_area =  cellpoly.intersection(pcellpoly).area/ cellpoly.area
                    pcell_overlap_area = pcellpoly.intersection(cellpoly).area/ pcellpoly.area
                    
                    if (cell_overlap_area > 0.5 or pcell_overlap_area > 0.5 and not assignment_dict[nuclei_ids[z_level][counter]]):
                        # print("here")
                        if(assignment_dict[nuclei_ids[z_level][counter]] == True):
                            merged_nuclei.append(assignment_dict[nuclei_ids[z_level][counter]])
                        DAPI_segmentation[z_level +1][DAPI_segmentation[z_level +1] == nuclei_ids[z_level+1][pcell]] = nuclei_ids[z_level][counter]
                        # print(z_level, nuclei_ids[z_level+1][pcell], nuclei_ids[z_level][counter])
                        nuclei_ids[z_level+1][pcell] = nuclei_ids[z_level][counter]
                        assignment_dict[nuclei_ids[z_level][counter]] = True #set to True as cell has now been assigned 
                    # else:
                    #     print(cell_overlap_area, pcell_overlap_area, nuclei_ids[z_level][counter], )
        print(merged_nuclei)
        for nuclei in merged_nuclei:
            
            DAPI_segmentation[DAPI_segmentation == nuclei] = 0

        
        sDB = segmentationTask.get_feature_database() # get the cell feature database (from segmentation task)
        currentCells = sDB.read_features(fov=0)
        print(currentCells)
        self.get_feature_database().write_features(currentCells, fov=0) #create a copy of cell feature database hdf5
        print("features written")
        #find nuclei IDs
        nuclei_dict = {}
        nuclei_cell_map = {}
        for z_level in range(len(DAPI_segmentation)): #len(DAPI_segmentation)
            #create rtree for all cells for this Z 
            cell_rtree = rindex.Index()
            #for each geometry, add to rtree
            cell_ids_str = [] #store geometry cell ids 
            geom_cell_boundaries = [] #store all cell geometry boundaries
            with self.dataSet.open_hdf5_file(
                'a', 'feature_data', self.get_analysis_name(), 0, 'features') as f:
                geom_index = 0
                for cellid in f['featuredata'].keys():
                    # print("cellid", cellid)
                    try: 
                        for geom in list(f['featuredata'][cellid]['zIndex_'+str(self.parameters['starting_z'] + z_level)]):
                            geom_bounds = list(f['featuredata'][cellid]['zIndex_'+str(self.parameters['starting_z'] + z_level)][geom]["coordinates"])[0]
                            
                            cell_ids_str.append(str(cellid + "_"+ geom))
                            geom_cell_boundaries.append(geom_bounds)
                            cell_rtree.insert(geom_index, Polygon(geom_bounds).bounds)
                            # print(Polygon(geom_bounds).bounds)
                            geom_index +=1
                    except: pass
            print("find cells")
            for i, nuclei in enumerate(micron_allz_DAPI_boundaries[z_level]):
                nuclei_poly = Polygon(nuclei)
                nuclei_bb = nuclei_poly.bounds
                # print(nuclei_bb)
                possible_cells = list(cell_rtree.intersection(nuclei_bb))
                # print(possible_cells)
                for cell in possible_cells:
                    
                    nuclei_id = nuclei_ids[z_level][i]
                    if(nuclei_poly.within(Polygon(geom_cell_boundaries[int(cell)]))):
                        #assign this nuclei to this geometry
                        cellid = cell_ids_str[int(cell)].split("_")[0]
                        nuclei_cell_map[nuclei_id] = cellid
                        # print(z_level, cellid, nuclei_id)
                        try:
                            nuclei_dict[nuclei_id][(self.parameters['starting_z'] + z_level)*1.5].append(Polygon(nuclei))
                        except:
                            try:
                                nuclei_dict[nuclei_id][(self.parameters['starting_z'] + z_level)*1.5] = [Polygon(nuclei)]
                            except:
                                nuclei_dict[nuclei_id] = {}
                                nuclei_dict[nuclei_id][(self.parameters['starting_z'] + z_level)*1.5] = [Polygon(nuclei)]


                    elif(nuclei_poly.intersection(Polygon(geom_cell_boundaries[int(cell)]))):
                        #mark this cell as problematic
                        with self.dataSet.open_hdf5_file(
                                        'a', 'feature_data', self.get_analysis_name(), 0, 'features') as f:
                            featureGroup = f['featuredata'][cellid]
                            featureGroup.attrs['problematic'] = True
                        #remove nuclei from DAPI masks 
                        DAPI_segmentation[DAPI_segmentation == int(cellid)] = 0
                    else:
                        #remove nuclei from DAPI masks as nuclei does not belong to any cell
                        DAPI_segmentation[DAPI_segmentation == int(cellid)] = 0

        #resave DAPI segmentation with removed cell ids 
        self.dataSet.save_numpy_analysis_result(
            np.array(DAPI_segmentation), 'segmentation',
            self.get_analysis_name(), resultIndex=0,
            subdirectory='DAPI')

        # print(nuclei_dict)
        print("saving metadata")
        for nuclei in nuclei_dict.keys():
            nucleipolygons = []
            z_levels = []
            #extract the geometries 
            #extract cell id, geometry
            cellid = nuclei_cell_map[nuclei]
            #extract polygons for each Z
            for z_level in nuclei_dict[nuclei].keys():
                nucleipolygons.append(nuclei_dict[nuclei][z_level])
                z_levels.append(z_level)
            #create a nucleiSpatialFeature object 
            nuclei_feature = spatialfeature.SpatialFeature(nucleipolygons, fov=0, zCoordinates = z_levels, uniqueID=str('nuclei_' + str(nuclei)))      
            #assign nuclei to its cell             
            with self.dataSet.open_hdf5_file(
            'a', 'feature_data', self.get_analysis_name(), 0, 'features') as f:
                featureGroup = f['featuredata'][cellid].create_group(str('nuclei_' + str(nuclei)))
                featureGroup.attrs['id'] = np.string_(str(nuclei_feature.get_feature_id()))
                featureGroup.attrs['bounding_box'] = np.array(nuclei_feature.get_bounding_box())
                featureGroup.attrs['volume'] = nuclei_feature.get_volume()
                featureGroup['DAPI_z_coords'] = nuclei_feature.get_z_coordinates()
                for i, bSet in zip(nuclei_feature.gets_z_coordinates(), nuclei_feature.get_boundaries()):
                    zBoundaryGroup = featureGroup.create_group('zIndex_' + str(int(i/1.5)))
                    for j, b in enumerate(bSet):
                        geometryGroup = zBoundaryGroup.create_group('np_' + str(j))
                        geometryDict = geometry.mapping(b)
                        geometryGroup.attrs['type'] = np.string_(geometryDict['type'])
                        geometryGroup['DAPI_coords'] = np.array(geometryDict['coordinates'])

class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata
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