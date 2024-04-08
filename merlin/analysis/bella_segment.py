import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Polygon, box
from rtree import index as rindex
import numpy as np
from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import  spatialfeature
from typing import List, Dict, Tuple
import zarr
from cellpose import utils
import matplotlib.pyplot as plt
import uuid
import datetime

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
            self.parameters['starting_z'] = 3
        if 'z_levels' not in self.parameters:
                    self.parameters['z_levels'] = [1,2,3,4,5,6]
        self.cellIDs = None
        self.cells_per_FOV = None

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
    def get_z_levels(self):
        return self.parameters["z_levels"]
    
    def are_lists_equal(self, list1, list2):
    # Helper function to check if two lists are equal
        return sorted(list1) == sorted(list2)

    def find_unique_lists(self, list_of_lists):
        unique_lists = []
        for lst in list_of_lists:
            if all(not self.are_lists_equal(lst, unique_lst) for unique_lst in unique_lists):
                unique_lists.append(lst)
        return unique_lists
    
    def _run_analysis(self) -> None:
        print("2D --> 3D")
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

        path_to_zarrs = self.parameters['path_to_segmentation_zarrs']
        zarr_prefix = self.parameters['zarr_prefix']
        filtered_seg_level = self.parameters['filter_seg']
        z_levels = [1,2,3,4,5,6]
        allz_boundaries = []
        positions = self.dataSet.get_stage_positions()
        minx = min(positions.X)
        miny = min(positions.Y)

        starting_z = int(self.parameters['starting_z'])
        z_levels = self.get_z_levels()
        middle_index = z_levels.index(starting_z)
        erosions = [None, "filtered_seg_0.01_eroded_1", "filtered_seg_0.1_eroded_1", "filtered_seg_0.3_eroded_1", \
                    "filtered_seg_0.3_eroded_1", "filtered_seg_0.3_eroded_1","filtered_seg_0.3_eroded_1" ]
        # print(minx)
        # print(miny)
        # print(positions)

        for zarr_level in z_levels: #z_levels
            # print(zarr_level)
            mask_label_matrix = zarr.open(path_to_zarrs + zarr_prefix +str(zarr_level) + ".zarr") [erosions[zarr_level]]
            boundaries = find_contours(np.transpose(mask_label_matrix), 0.9, fully_connected='high')
            shifted_boundaries = [np.asarray([[((coord[0]*2)*0.107 + minx), (coord[1]*2)*0.107 + miny] for coord in bound]) for bound in boundaries]
            allz_boundaries.append(shifted_boundaries)

        # xcoords = []
        # ycoords = []

        #testing on cellpose output until we can get lsd to work on single FOVs (for testing purposes) 
        # for z in z_levels: 
        #     mask_label_matrix = np.load(path_to_zarrs +  "2fov_bigtiff_stitch" + str(z) + "_seg.npy", allow_pickle=True)
        #     np.save("mask_label_matrix.npy", np.asarray([bound.tolist() for bound in utils.outlines_list(mask_label_matrix.item()['masks'])], dtype=object), allow_pickle=True)
        #     boundaries = utils.outlines_list(mask_label_matrix.item()['masks'])
        #     # print(len(boundaries))
        #     shifted_boundaries = [np.asarray([[((coord[0]*1)*0.107 + minx), (coord[1]*1)*0.107 + miny] for coord in bound]) for bound in boundaries]
        #     # unique_shifted_boundaries = self.find_unique_lists(shifted_boundaries)
        #     # for bound in shifted_boundaries: 
        #     #     for coord in bound: 
        #     #         xcoords.append(coord[0])
        #     #         ycoords.append(coord[1])
        #     allz_boundaries.append(shifted_boundaries)
        # plt.scatter(xcoords, ycoords, s= 0.1)
        # plt.savefig("test.png")
        
        ### COMBINE 2D CELLS INTO 3D CELLS ###
        #remove lines from segmentation:
        nonzero_boundaries = []

        for z in range(len(z_levels)):
            counter = 0
            nonzero_z_boundaries = []
            for cell in allz_boundaries[z]:
                if Polygon(cell).area > 1 and Polygon(cell).is_valid:
                    nonzero_z_boundaries.append(cell)
                counter +=1
            nonzero_boundaries.append(nonzero_z_boundaries)
        allz_boundaries = nonzero_boundaries
    
        z_cell_IDs = []
        counter = 0 
        for z in range(len(z_levels)):
            z_cellIDs = []
            for i in range(len(allz_boundaries[z])):
                z_cellIDs.append(uuid.uuid4().int)
            # counter += len(allz_boundaries[z])
            z_cell_IDs.append(z_cellIDs)

        cell_IDs = {}
        counter = 0
        for i in z_cell_IDs[middle_index]:
            cell_IDs[i] = []
            cell_IDs[i].append(allz_boundaries[middle_index][counter].tolist())    
            
            counter +=1 

        for i in range(middle_index, 0, -1):
            cell_num = 0
            print(i)
            above_rtree = rindex.Index()
            ii = 0 
            for cell in allz_boundaries[i-1]:
                above_rtree.insert(ii, Polygon(cell).bounds)
                ii +=1
                
            for cell in allz_boundaries[i]:
                cellpoly = Polygon(cell)
                possible_cells_below = above_rtree.intersection(cellpoly.bounds)
        #         print(list((possible_cells_below)))
                for pcell in np.unique(list((possible_cells_below))):
                    pcellpoly = Polygon(allz_boundaries[i-1][pcell])
                    cell_overlap_area =  cellpoly.intersection(pcellpoly).area/ cellpoly.area
                    pcell_overlap_area = pcellpoly.intersection(cellpoly).area/ pcellpoly.area
                    if (cell_overlap_area > 0.2 and pcell_overlap_area > 0.5):
                        if z_cell_IDs[i][cell_num] in cell_IDs.keys():
                            cell_IDs[z_cell_IDs[i][cell_num]].append(allz_boundaries[i-1][pcell].tolist())
                        else: 
                            emptylist = []
                            emptylist.append(allz_boundaries[i][cell_num].tolist())
                            cell_IDs[z_cell_IDs[i][cell_num]] = emptylist
                            cell_IDs[z_cell_IDs[i][cell_num]].append(allz_boundaries[i-1][pcell].tolist())
                        z_cell_IDs[i-1][pcell] = z_cell_IDs[i][cell_num]

                cell_num +=1
                    
        for i in range(middle_index, len(z_levels) -1):
            # print(i)
            cell_num = 0 
            below_rtree = rindex.Index()
            ii = 0 
            for cell in allz_boundaries[i+1]:
                below_rtree.insert(ii, Polygon(cell).bounds)
                ii+=1
                
            for cell in allz_boundaries[i]:
                cellpoly = Polygon(cell)
                possible_cells_below = below_rtree.intersection(cellpoly.bounds)
                for pcell in np.unique(list((possible_cells_below))):
                    pcellpoly = Polygon(allz_boundaries[i+1][pcell])
                    cell_overlap_area =  cellpoly.intersection(pcellpoly).area/ cellpoly.area
                    pcell_overlap_area = pcellpoly.intersection(cellpoly).area/ pcellpoly.area
                    
                    if (cell_overlap_area > 0.2 and pcell_overlap_area > 0.5):
                        if z_cell_IDs[i][cell_num] in cell_IDs.keys():
                            cell_IDs[z_cell_IDs[i][cell_num]].append(allz_boundaries[i+1][pcell].tolist())
                        else: 
                            emptylist = []
                            emptylist.append(allz_boundaries[i][cell_num].tolist())
                            cell_IDs[z_cell_IDs[i][cell_num]] = emptylist
                            cell_IDs[z_cell_IDs[i][cell_num]].append(allz_boundaries[i+1][pcell].tolist())
                        z_cell_IDs[i+1][pcell] = z_cell_IDs[i][cell_num]
        #             else:
        #                 cell_IDs[z_cell_IDs[i+1][pcell]] = allz_boundaries[i+1][pcell].tolist()
                cell_num +=1

        for z in range(0,6):
            counter = 0
            for cellid in z_cell_IDs[z]:
                if cellid not in cell_IDs.keys():
                    emptylist =[]
                    emptylist.append(allz_boundaries[z][counter])
                    cell_IDs[cellid] = emptylist
                counter +=1

        ### ASSIGN CELLS TO FOVS ###

        fov_rtree = rindex.Index()
        fov_bounds = []
        fov_num = 0
        fov_cell_dict= {}
        for fov in range(len(positions)):
            # print(positions.loc[fov][0])
            fov_box = box(positions.loc[fov][0], positions.loc[fov][1], positions.loc[fov][0]+220, positions.loc[fov][1] + 220)
            fov_rtree.insert(fov, fov_box.bounds)
            # print(fov_box.bounds)
            fov_bounds.append(fov_box)
            fov_cell_dict[fov] = {}
            # fov_num +=1

        for z in range(len(z_levels)):
            counter = 0
            for cell in allz_boundaries[z]: 
                cellpoly = Polygon(cell)
                cell_bounding_box = cellpoly.bounds
                # print(cell_bounding_box)
                if(cellpoly.area > 0 ):
                    possible_FOVs = fov_rtree.intersection(cell_bounding_box)
                    for FOV in list(possible_FOVs): 
                        if(cellpoly.intersects(fov_bounds[FOV]) != None):
                            curr_cellid = z_cell_IDs[z][counter]
                            actualZ = z+1 #TODO change this!!!!
                            if curr_cellid not in fov_cell_dict[FOV].keys():
                                fov_cell_dict[FOV][curr_cellid] = [{actualZ: cellpoly}]
                            else:
                                fov_cell_dict[FOV][curr_cellid].append({actualZ: cellpoly})
                            # fov_cell_dict[FOV].append({curr_cellid: [z, cellpoly]}) # add cell id to fov's list of cells
                counter +=1

        
        featureDB = self.get_feature_database()
        for fov in range(len(positions)):
            # fov_cell_dict[fov] = list(set(fov_cell_dict[fov]))
            #get all z levels for each cell in this fov

            featurelist = []
            for cellid in fov_cell_dict[fov]: 
                z_levels = []
                cellpolygons = []
                for subcell in fov_cell_dict[fov][cellid]:
                    z_levels.append(list(subcell.keys())[0]+1.5) #TODO CHECK THIS WORKS 
                    cellpolygons.append([list(subcell.values())[0]])
                # create spatial feature object
                # print(cellpolygons)
                # print (z_levels)
                featurelist.append(spatialfeature.SpatialFeature(cellpolygons, fov=fov, zCoordinates = z_levels, uniqueID=cellid))
                # print(cellid)
            featureDB.write_features(featurelist, fov)

class KolabCombineCells(FeatureSavingAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        
        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        if 'starting_z' not in self.parameters:
            self.parameters['starting_z'] = 4 
        if "z_levels" not in self.parameters: 
            self.parameters["z_levels"] = [1,2,3,4,5,6]

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def get_z_levels(self):
        return self.parameters["z_levels"]
    
    def remove_interior_boundaries(self, fragmentIndex):

        cells = self.segmentTask.get_feature_database()\
                        .read_features(fragmentIndex)
        
        

        valid_cells = [] 
        for z in range(len(self.get_z_levels())):
            #get cells for each z level
            z_cells = []

            for cell in cells:
                print(cell._zCoordinates)
                if cell._zCoordinates == z+1:
                    z_cells.append(cell)
            
            temp_rtree = rindex.Index()
            i=0
            for cell in z_cells:
                # cellboundary_coords = cell.get_boundaries()
                temp_rtree.insert(i, cell.get_bounding_box(), obj = str(i))
                i+=1
            
            # valid_cells = []
            # print(len(z_cells))
            i =0
            for cell in z_cells:
                intersecting_bb = temp_rtree.intersection(cell.get_bounding_box())
                possible_intersecting_cells = list(intersecting_bb)
                possible_intersecting_cells.remove(i) #remove self
                j=0
                for pcell in np.unique(possible_intersecting_cells):
                    if cell.is_contained_within_boundary(z_cells[pcell]):
                        j+=1
                if j==0:
                    valid_cells.append(cell)
                i +=1
        # self._reset_analysis(fragmentIndex)    
        self.get_feature_database().write_features(valid_cells, fragmentIndex)

    def _run_analysis(self, fragmentIndex) -> None:
        pass
        #remove cells inside other cells 
        # self.remove_interior_boundaries(fragmentIndex)


        # starting_z = self.parameters['starting_z'] 
        # z_levels = self.get_z_levels()
        # middle_index = z_levels.index(starting_z)
    
        # # find cell id's of cells in the middle z level
        # fov_cells = self.get_feature_database().read_features(fragmentIndex)
        
        # z_cells = []

        # cellIDs = []
        # for z in range(0, len(self.get_z_levels()) + 1):
        #     z_cells.append([])
        #     cellIDs.append([])

        # for cell in fov_cells: 
        #     #find which z level this cell is in 
        #     # print(cell._zCoordinates)
        #     z_cells[cell._zCoordinates[0]].append(cell)
        #     cellIDs[cell._zCoordinates[0]].append(cell._uniqueID)
        #     # print(cell._uniqueID)

        # starting_cellIDs = {}
        # counter = 0
        # for cellid in cellIDs[middle_index]:
        #     starting_cellIDs[cellid] = {}
        #     starting_cellIDs[cellid][0] = [] #store polygon objects
        #     starting_cellIDs[cellid][1] = [] #store corresponding z levels 
        #     starting_cellIDs[cellid][0].append(z_cells[middle_index][counter])
        #     starting_cellIDs[cellid][1].append(middle_index)
        #     counter +=1

        # # print(len(fov_cells))
        #     # print(cell._uniqueID)
        # # print("original" , cellIDs)
        # # print()
        # # find all shared cells in the z level above the starting z level
        
        # for z in range(middle_index, 0, -1):
            
        #     cell_num = 0
        #     above_rtree = rindex.Index()

        #     ii = 0
        #     for cell in z_cells[z-1]:
        #         above_rtree.insert(ii, cell.get_bounding_box(), obj = str(ii))
        #         ii +=1
                
        #     for cell in z_cells[z]:
        #         possible_cells_below = above_rtree.intersection(cell.get_bounding_box())
        #         for pcell in np.unique(list((possible_cells_below))):
        #             pcell = z_cells[z-1][pcell]

        #             cell_overlap_area =  cell.intersection(pcell)
        #             pcell_overlap_area = pcell.intersection(cell)

        #             if (cell_overlap_area > 0.2 and pcell_overlap_area > 0.5):
        #                 #create a new spatial feature with the same uniqueID as the cell above
        #                 spatialfeature.SpatialFeature([[cellpoly]], fragmentIndex, [z_levels[z]], uniqueID= cell._uniqueID)
        #                 starting_cellIDs[cell._uniqueID].append(pcell)
        #                 # pcell._uniqueID = z_cells[z][cell_num]._uniqueID
        #         cell_num +=1
                
        # for z in range(middle_index, len(z_levels) -1):
        #     cell_num = 0 
        #     below_rtree = rindex.Index()
        #     ii = 0 
        #     for cell in z_cells[z+1]:
        #         below_rtree.insert(ii, cell.get_bounding_box(), obj = str(ii))
        #         ii+=1
                
        #     for cell in z_cells[z]:
        #         # cellpoly = Polygon(cell)
        #         possible_cells_below = below_rtree.intersection(cell.get_bounding_box())
        #         for pcell in np.unique(list((possible_cells_below))):
        #             pcell = z_cells[z+1][pcell]
        #             cell_overlap_area =  cell.intersection(pcell)
        #             pcell_overlap_area = pcell.intersection(cell)
                    
        #             if (cell_overlap_area > 0.2 and pcell_overlap_area > 0.5):
        #                 pcell._uniqueID = cell._uniqueID
        #         cell_num +=1

        # _3d_cells = []

        # self.segmentTask.get_feature_database().read_features(fragmentIndex)
        self.get_feature_database().write_features(self.segmentTask.get_feature_database().read_features(fragmentIndex), fragmentIndex)

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