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

class PartitionBarcodesDAPI(analysistask.AnalysisTask):

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
                self.parameters['assignment_task'], 
                self.parameters['DAPI_segmentation_task']]
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
        # path_to_zarrs = self.parameters['path_to_segmentation_zarrs']
        # zarr_name = self.parameters['zarr_name']
        scale_factor=self.parameters['scale_factor'] #should be 4 for S2, 2 for s1, and 1 for s0.

        eroded = self.parameters['DAPI_eroded_zarr']

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()

        allBarcodes = bcDB.get_barcodes(columnList=["barcode_id", "global_x", "global_y", "z"])
        
        #load the scaling factor 
        microns_per_pixel = self.dataSet.get_microns_per_pixel()
        positions = self.dataSet.get_stage_positions()

        minx_microns = min(positions.X)
        miny_microns = min(positions.Y)

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
        numz=len(zarr.open(eroded))
        #print(os.path.join(path_to_zarrs, zarr_name))
        #print(numz) 

        #dictionary storing transcript locations for each cell 
        cell_transcript_locations = {} #key is cell id, value is a list of tuples of the transcript locations for that cell

        # partitioned_transcripts = pandas.DataFrame(columns= ['nuclear_id', 'barcode_id', 'global_x', 'global_y', 'z'])
        nuclei_id_list = []
        barcode_id_list = []
        global_x_list = []
        global_y_list = []
        z_list = []


        # unique_values=np.unique(DAPI_masks)
        # num_rows = len(unique_values) #number of cells 
        # num_columns = len(np.unique(txs_scaled['barcode_id'])) # number of genes,

        # #initialize the cell by gene matrix 
        # column_names = ['nuclei'] + [str(i) for i in range(num_columns)]
        # data = [[int(0)] * len(column_names) for _ in range(num_rows)]
        # nxg = pandas.DataFrame(data, columns=column_names)
        # nxg['nuclei'] = unique_values


        for z_index in range(numz): 

            print("working on z index ", z_index)
            
            # masks = rescale(zarr.open(os.path.join(path_to_zarrs, zarr_name))[z_index,:,:], [scale_factor, scale_factor], order=0) #load in the zarr file of the masks file for the given z 
            # #contain pixel wise labelling of which pixel is a cell, which cell it is, and which ones are not 
            DAPI_masks = rescale(zarr.open(f"{eroded}/3D")[z_index,:,:], [scale_factor, scale_factor], order=0)

            xmax = DAPI_masks.shape[1]
            ymax = DAPI_masks.shape[0]

            unique_values=np.unique(DAPI_masks)
            num_rows = len(unique_values) #number of cells 
            num_columns = len(np.unique(txs_scaled['barcode_id'])) # number of genes,

            #initialize the cell by gene matrix 
            column_names = ['nuclei'] + [str(i) for i in range(num_columns)]
            data = [[int(0)] * len(column_names) for _ in range(num_rows)]
            nxg = pandas.DataFrame(data, columns=column_names)
            nxg['nuclei'] = unique_values

            # report memory usage
            print(f"Total memory usage of the DataFrame (including index): {nxg.memory_usage(deep=True, index=True).sum()} bytes")

            z_txs_scaled = txs_scaled[txs_scaled['z'] == z_index+self.parameters['starting_z']]

            for index, row in z_txs_scaled.iterrows():
                z = int(row['z'])
                if True:
                    x = math.ceil(row['global_x']) if row['global_x'] % 1 >= 0.444445 else math.floor(row['global_x'])
                    y =  math.ceil(row['global_y']) if row['global_y'] % 1 >= 0.444445 else math.floor(row['global_y'])
                    #print("matching Z found") #there are matching Zs being found, but none of them are fulfilling the following if statement 
                    #print(y) #the y values are all negative and this is what is eliminating them from fulfilling the next statement 

                    #previous code 11/16 that is not allowing the barcodes to pass into the loop that actually does the partitioning because they don't pass the 
                    #requirement to be positive values 
                    if x >= 0 and y >= 0 and x < xmax and y < ymax: # check to see if this is the issue by seeing what actually gets compared 
                        #print("true1")
                        nuclei_index = DAPI_masks[y, x] # indexing is flipped - first index is y in images
                        if nuclei_index:
                            #print("ADDED TO CXG MATRIX")
                            if not index%10000:
                                print(index) #report progress
                            nxg.loc[nxg['nuclei'] == nuclei_index, str(int(row['barcode_id']))] += 1
                            
                            nuclei_id_list.append(int(nuclei_index))
                            barcode_id_list.append(int(row['barcode_id']))
                            global_x_list.append(float((row['global_x']+minx)*microns_per_pixel))
                            global_y_list.append(float((row['global_y']+miny)*microns_per_pixel))
                            z_list.append(float(row['z']))
                            # new_data = pandas.DataFrame({'cell_id': int(cell_index), 'nuclear_id': int(nuclei_index), 'barcode_id': int(row['barcode_id']), 'global_x': float(row['global_x']), 'global_y': float(row['global_y']), 'z': float(row['z'])},index=[0])
                            # partitioned_transcripts = pandas.concat([partitioned_transcripts, new_data], ignore_index=True).astype(float)

            
            z_name=z_index+self.parameters['starting_z']
            nxg_filename = f'nxg_z{z_name}'
            self.dataSet.save_dataframe_to_csv(dataframe=nxg,resultName=nxg_filename,analysisTask=self.get_analysis_name())

            

            #cxg.to_csv(cxg_filename, index=False) #where do you want the csv file written to? currently just written to whatever the working directory of this function will be 
        partitioned_transcripts = pandas.DataFrame({'nuclear_id': nuclei_id_list, 'barcode_id': barcode_id_list, 'global_x': global_x_list, 'global_y': global_y_list, 'z': z_list}).astype(float)
        self.dataSet.save_dataframe_to_csv(dataframe=partitioned_transcripts,resultName='DAPI_partitioned_txs',analysisTask=self.get_analysis_name())
        
        sDB = assignmentTask.get_feature_database() # get the feature database (from segmentation task)
        currentNuclei = sDB.read_features()
        for nucleus in currentNuclei:
            nuclearID = nucleus._uniqueID
            nuclear_txs = partitioned_transcripts[partitioned_transcripts["nuclear_id"] == nuclearID]
            nucleus.set_transcript_list(nuclear_txs)

        self.get_feature_database().write_feature_transcripts(currentNuclei, 0) # write the transcripts for each cell to the feature database (hdf5 object)

        #load in all of the cell x gene matrices and then concatenate them into a combined one 
        dataframes = []
        #read in each data frame and store them in the dataframes list 
        for z_index in range(numz):
            z_name=z_index+self.parameters['starting_z']
            file_name = f'nxg_z{z_name}'
            #df = pandas.read_csv(file_name)
            df=self.dataSet.load_dataframe_from_csv(resultName=file_name,analysisTask=self.get_analysis_name())
            dataframes.append(df)
        concatenated_df = pandas.concat(dataframes, ignore_index=True) #concatenate the dataframes 
        merged_df = concatenated_df.groupby('nuclei', as_index=False).sum() #merge the rows from different z values that belong to the same cell in the cxg matrix 
        self.dataSet.save_dataframe_to_csv(dataframe=merged_df,resultName='nxg_combined',analysisTask=self.get_analysis_name())
       
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