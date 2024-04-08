import pandas
import numpy as np
import os 
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.core import dataset
import datetime
import zarr 
from skimage.transform import rescale
import math 

class PartitionBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.
    """
    

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['filter_task'],
                self.parameters['assignment_task']]
                #self.parameters['alignment_task']
    
    def get_partitioned_barcodes(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            # Concatenate all the data frames containing the partitioned barcodes from each fov together
            concatenated_df = pandas.concat(
                [self.get_partitioned_barcodes(fov)
                 for fov in self.dataSet.get_fovs()])
            
            return concatenated_df.groupby(concatenated_df.index).sum() # sum the counts for each barcode across all fovs for each cell id

        return self.dataSet.load_dataframe_from_csv(
            'counts_per_cell', self.get_analysis_name(), fov, index_col=0)
    
    
    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)
    
    def _run_analysis(self):
        print("partition begin")
        print(os.getcwd())
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        # alignTask = self.dataSet.load_analysis_task(
        #     self.parameters['alignment_task'])
        if 'starting_z' not in self.parameters:
            self.parameters['starting_z'] = 2
        path_to_zarrs = self.parameters['path_to_segmentation_zarrs']
        zarr_name = self.parameters['zarr_name']
        s_scaling_factor=self.parameters['s_scaling_factor'] #should be 4 for S2, 2 for s1, and 1 for s0.

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()

        allBarcodes = bcDB.get_barcodes(columnList=["barcode_id", "global_x", "global_y", "z"])
        
        #load the scaling factor 
        microns_per_pixel = self.dataSet.get_microns_per_pixel()
        positions = self.dataSet.get_stage_positions()
        minx = min(positions.X)/microns_per_pixel 
        miny = min(positions.Y)/microns_per_pixel 
        
        #load the transcript data 
        allBarcodes = allBarcodes.reset_index().copy(deep=True)
        txs_scaled=allBarcodes.copy(deep=True)
        
        #scale the transcript positions based on the scaling factor 
        txs_scaled['global_x'] /= microns_per_pixel 
        txs_scaled['global_x'] -= minx
        txs_scaled['global_y'] /= microns_per_pixel
        txs_scaled['global_y'] -= miny

        #perform partition for each z at a time 
        numz=len(zarr.open(os.path.join(path_to_zarrs, zarr_name)))
        #print(os.path.join(path_to_zarrs, zarr_name))
        #print(numz) 

        #dictionary storing transcript locations for each cell 
        cell_transcript_locations = {} #key is cell id, value is a list of tuples of the transcript locations for that cell


        for z_index in range(numz): 

            print("working on z index ", z_index)
            
            masks = rescale(zarr.open(os.path.join(path_to_zarrs, zarr_name))[z_index,:,:], [s_scaling_factor, s_scaling_factor], order=0) #load in the zarr file of the masks file for the given z 
            #contain pixel wise labelling of which pixel is a cell, which cell it is, and which ones are not 

            xmax = masks.shape[1]
            ymax = masks.shape[0]

            unique_values=np.unique(masks)
            num_rows = len(unique_values) #number of cells 
            num_columns = len(np.unique(txs_scaled['barcode_id'])) # number of genes,

            #initialize the cell by gene matrix 
            column_names = ['cell'] + [str(i) for i in range(num_columns)]
            data = [[int(0)] * len(column_names) for _ in range(num_rows)]
            cxg = pandas.DataFrame(data, columns=column_names)
            cxg['cell'] = unique_values

            # report memory usage
            print(f"Total memory usage of the DataFrame (including index): {cxg.memory_usage(deep=True, index=True).sum()} bytes")

            for index, row in txs_scaled.iterrows():
                z = int(row['z'])
                if z == z_index+self.parameters['starting_z']:
                    x = math.ceil(row['global_x']) if row['global_x'] % 1 >= 0.444445 else math.floor(row['global_x'])
                    y =  math.ceil(row['global_y']) if row['global_y'] % 1 >= 0.444445 else math.floor(row['global_y'])
                    #print("matching Z found") #there are matching Zs being found, but none of them are fulfilling the following if statement 
                    #print(y) #the y values are all negative and this is what is eliminating them from fulfilling the next statement 

                    #previous code 11/16 that is not allowing the barcodes to pass into the loop that actually does the partitioning because they don't pass the 
                    #requirement to be positive values 
                    if x >= 0 and y >= 0 and x < xmax and y < ymax:#check to see if this is the issue by seeing what actually gets compared 
                        #print("true1")
                        cell_index = masks[y, x] # indexing is flipped - first index is y in images
                        if cell_index:
                            #print("ADDED TO CXG MATRIX")
                            if not index%10000:
                                print(index) #report progress
                            cxg.loc[cxg['cell'] == cell_index, str(int(row['barcode_id']))] += 1
                            # if(cell_index in cell_transcript_locations):
                            #     cell_transcript_locations[cell_index].append([row['barcode_id'],allBarcodes.iloc[index]['global_x'], allBarcodes.iloc[index]['global_y'],z])
                            # else:
                            #     cell_transcript_locations[cell_index]=[[row['barcode_id'],allBarcodes.iloc[index]['global_x'], allBarcodes.iloc[index]['global_y'],z]]
                '''
                    #the code below also throws an error because I think the normalization to pixels didn't quite work 
                    cell_index = masks[y, x] # indexing is flipped - first index is y in images
                    if cell_index: #if the cell index pulled from the mask isn't a null, then add the transcript to the cxg matrix 
                         print("ADDED TO CXG MATRIX")
                         if not index%10000:
                            print(index) #report progress
                         cxg.loc[cxg['cell'] == cell_index, str(int(row['barcode_id']))] += 1
                         '''
            z_name=z_index+self.parameters['starting_z']
            cxg_filename = f'cxg_z{z_name}'
            self.dataSet.save_dataframe_to_csv(dataframe=cxg,resultName=cxg_filename,analysisTask=self.get_analysis_name())
            #cxg.to_csv(cxg_filename, index=False) #where do you want the csv file written to? currently just written to whatever the working directory of this function will be 

        #save the transcript locations for each cell to hdf5 file
        # sDB = assignmentTask.get_feature_database()
        # currentCells = sDB.read_features(0)
        # print(currentCells)
        # currentCells_dict = {}
        # for cell in currentCells:
        #     # print(cell.get_feature_id())
        #     currentCells_dict[cell.get_feature_id()] = cell
        # for cell in cell_transcript_locations.keys():
        #     try:
        #         currentCells_dict[cell].set_transcript_list(cell_transcript_locations[cell])
        #     except: pass
        # # currentCells[pcell].set_transcript_list(transcriptdf)
        # self.get_feature_database().write_feature_transcripts(currentCells, 0)

        #load in all of the cell x gene matrices and then concatenate them into a combined one 
        dataframes = []
        #read in each data frame and store them in the dataframes list 
        for z_index in range(numz):
            z_name=z_index+self.parameters['starting_z']
            file_name = f'cxg_z{z_name}'
            #df = pandas.read_csv(file_name)
            df=self.dataSet.load_dataframe_from_csv(resultName=file_name,analysisTask=self.get_analysis_name())
            dataframes.append(df)
        concatenated_df = pandas.concat(dataframes, ignore_index=True) #concatenate the dataframes 
        merged_df = concatenated_df.groupby('cell', as_index=False).sum() #merge the rows from different z values that belong to the same cell in the cxg matrix 
        self.dataSet.save_dataframe_to_csv(dataframe=merged_df,resultName='cxg_combined',analysisTask=self.get_analysis_name())
       
        print("partition finish")
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

class ExportPartitionedBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that combines counts per cells data from each
    field of view into a single output file.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['partition_task']]

    def _run_analysis(self):
        pTask = self.dataSet.load_analysis_task(
                    self.parameters['partition_task'])
        parsedBarcodes = pTask.get_partitioned_barcodes()
        self.dataSet.save_dataframe_to_csv(
                    parsedBarcodes, 'barcodes_per_feature',
                    self.get_analysis_name())