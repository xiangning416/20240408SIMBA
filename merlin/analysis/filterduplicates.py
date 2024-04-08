from merlin.core import analysistask
from merlin.util import barcodedb
import numpy as np 
import pandas as pd 
import shapely
from shapely.geometry import box


class FilterDuplicates(analysistask.AnalysisTask):

    """
    An analysis task that filters duplicates from the barcodes.csv and positions.csv files and writes a new hdf5 file structure 
    """


    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'columns' not in self.parameters:
            self.parameters['columns'] = ['barcode_id', 'global_x',
                                          'global_y', 'cell_index']
        if 'exclude_blanks' not in self.parameters:
            self.parameters['exclude_blanks'] = True

        self.columns = self.parameters['columns']
        self.excludeBlanks = self.parameters['exclude_blanks']

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        return barcodedb.PyTablesBarcodeDB(self.dataSet,self)

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['filter_task']]

    def calculate_overlap(self,square1, square2, side_length=220): #TODO Need to change the side length from 220 to a parameter 
        # Create shapely box objects for both squares
        # box(minx, miny, maxx, maxy)
        square1_box = box(square1[0], square1[1], square1[0] + side_length, square1[1] + side_length)
        square2_box = box(square2[0], square2[1], square2[0] + side_length, square2[1] + side_length)

        # Calculate the intersection (overlap) of the two squares
        intersection = square1_box.intersection(square2_box)

        # Check if they actually intersect
        if intersection.is_empty:
            return None, 0
        else:
            return intersection.bounds

    def _run_analysis(self):
        #reading in the barcode data as a pandas dataframe 
        filterTask = self.dataSet.load_analysis_task(
                    self.parameters['filter_task'])        

        transcript_file = filterTask.get_barcode_database().get_barcodes(
                columnList=self.columns)
        transcript_file=transcript_file.reset_index(drop=True)

        print(transcript_file.head)
        #reading in the position list 
        position_file= self.dataSet.get_stage_positions()
        position_file=position_file.rename(columns={0: "x", 1: "y"})

        #put in the filtering duplicates function 
        # Store indices to remove later
        indices_to_remove = []

        # Iterate over positions

        
        for i in range(position_file.shape[0]):
            for j in range(i):
                square1_top_left = (position_file.iloc[i, 0], position_file.iloc[i, 1])
                #print(square1_top_left)
                square2_top_left = (position_file.iloc[j, 0], position_file.iloc[j, 1])
                overlap_region = self.calculate_overlap(square1_top_left, square2_top_left)
                if overlap_region[0] is not None:
                    # Instead of editing the transcript_file, accumulate the indices
                    indices = transcript_file[
                        (transcript_file['fov'] == i) &
                        (transcript_file['global_x'] >= overlap_region[0]) & 
                        (transcript_file['global_x'] <= overlap_region[2]) & 
                        (transcript_file['global_y'] >= overlap_region[1]) & 
                        (transcript_file['global_y'] <= overlap_region[3])
                    ].index
                    indices_to_remove.extend(indices)
            
                    
        # Once all indices are collected, remove them from the DataFrame
        #transcript_file.to_csv("D:/transcripts.csv")
        #np.savetxt("D:/transcripts.csv", transcript_file, delimiter=",", fmt='%d')
        #np.savetxt("D:/test.csv", indices_to_remove, delimiter=",", fmt='%d')
        #print(position_file.head)
        #print(transcript_file.head)
        #print(transcript_file.shape)
        #print(len(np.unique(indices_to_remove)))
        mask = ~transcript_file.index.isin(indices_to_remove)
        unique, counts = np.unique(mask, return_counts=True)
        #print(unique)
        #print(counts)
        output = transcript_file[mask]

        
        #print(output.shape)

        print(output.columns)

        output=transcript_file
        #write the filtered dataset to the hdf5 file 
        self.get_barcode_database().write_barcodes(output)

        #write csv file to validate hdf5 output 
        self.dataSet.save_dataframe_to_csv(output, 'duplicate_filtered_barcodes', self,
                                           index=False)
