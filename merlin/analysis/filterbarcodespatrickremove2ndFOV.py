import numpy as np
import pandas
import os
from scipy import optimize
import shapely
from shapely.geometry import box
from merlin.core import analysistask
from merlin.analysis import decode
from merlin.util import barcodedb
import pandas as pd

#this file will modify the filterbarcodespatrick file to remove the duplicates from the frame with less transcripts in that region 

class AbstractFilterBarcodes(decode.BarcodeSavingParallelAnalysisTask):
    """
    An abstract class for filtering barcodes identified by pixel-based decoding.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_codebook(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        return decodeTask.get_codebook()


class FilterBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 3
        if 'intensity_threshold' not in self.parameters:
            self.parameters['intensity_threshold'] = 200
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 1e6

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        return barcodedb.PyTablesBarcodeDB(self.dataSet,self)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

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

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        areaThreshold = self.parameters['area_threshold']
        intensityThreshold = self.parameters['intensity_threshold']
        distanceThreshold = self.parameters['distance_threshold']
        barcodeDB = self.get_barcode_database()

        #reading in the position list 
        position_file= self.dataSet.get_stage_positions()
        position_file=position_file.rename(columns={0: "x", 1: "y"})

        #reading in the barcodes file 
        transcript_file=decodeTask.get_barcode_database().get_filtered_barcodes(
                areaThreshold, intensityThreshold,
                distanceThreshold=distanceThreshold, fov=fragmentIndex)
        transcript_file=transcript_file.reset_index(drop=True) #check to see if resetting the indices affects merlin downstream 

        
        barcodeDB.write_barcodes(
            transcript_file,
            fov=fragmentIndex)



class GenerateAdaptiveThreshold(analysistask.AnalysisTask):

    """
    An analysis task that generates a three-dimension mean intenisty,
    area, minimum distance histogram for barcodes as they are decoded.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'tolerance' not in self.parameters:
            self.parameters['tolerance'] = 0.001
        # ensure decode_task is specified
        decodeTask = self.parameters['decode_task']

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 1800

    def get_dependencies(self):
        return [self.parameters['run_after_task']]

    def get_blank_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('blank_counts', self)

    def get_coding_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('coding_counts', self)

    def get_total_count_histogram(self) -> np.ndarray:
        return self.get_blank_count_histogram() \
               + self.get_coding_count_histogram()

    def get_area_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('area_bins', self)

    def get_distance_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'distance_bins', self)

    def get_intensity_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'intensity_bins', self, None)

    def get_blank_fraction_histogram(self) -> np.ndarray:
        """ Get the normalized blank fraction histogram indicating the
        normalized blank fraction for each intensity, distance, and area
        bin.

        Returns: The normalized blank fraction histogram. The histogram
            has three dimensions: mean intensity, minimum distance, and area.
            The bins in each dimension are defined by the bins returned by
            get_area_bins, get_distance_bins, and get_area_bins, respectively.
            Each entry indicates the number of blank barcodes divided by the
            number of coding barcodes within the corresponding bin
            normalized by the fraction of blank barcodes in the codebook.
            With this normalization, when all (both blank and coding) barcodes
            are selected with equal probability, the blank fraction is
            expected to be 1.
        """
        blankHistogram = self.get_blank_count_histogram()
        totalHistogram = self.get_coding_count_histogram()
        blankFraction = blankHistogram / totalHistogram
        blankFraction[totalHistogram == 0] = np.finfo(blankFraction.dtype).max
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankFraction /= blankBarcodeCount/(
                blankBarcodeCount + codingBarcodeCount)
        return blankFraction

    def calculate_misidentification_rate_for_threshold(
            self, threshold: float) -> float:
        """ Calculate the misidentification rate for a specified blank
        fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The estimated misidentification rate, estimated as the
            number of blank barcodes per blank barcode divided
            by the number of coding barcodes per coding barcode.
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()

        selectBins = blankFraction < threshold
        codingCounts = np.sum(codingHistogram[selectBins])
        blankCounts = np.sum(blankHistogram[selectBins])

        return ((blankCounts/blankBarcodeCount) /
                (codingCounts/codingBarcodeCount))

    def calculate_threshold_for_misidentification_rate(
            self, targetMisidentificationRate: float) -> float:
        """ Calculate the normalized blank fraction threshold that achieves
        a specified misidentification rate.

        Args:
            targetMisidentificationRate: the target misidentification rate
        Returns: the normalized blank fraction threshold that achieves
            targetMisidentificationRate
        """
        tolerance = self.parameters['tolerance']
        def misidentification_rate_error_for_threshold(x, targetError):
            return self.calculate_misidentification_rate_for_threshold(x) \
                - targetError
        return optimize.newton(
            misidentification_rate_error_for_threshold, 0.2,
            args=[targetMisidentificationRate], tol=tolerance, x1=0.3,
            disp=False)

    def calculate_barcode_count_for_threshold(self, threshold: float) -> float:
        """ Calculate the number of barcodes remaining after applying
        the specified normalized blank fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The number of barcodes passing the threshold.
        """
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()
        return np.sum(blankHistogram[blankFraction < threshold]) \
            + np.sum(codingHistogram[blankFraction < threshold])

    def extract_barcodes_with_threshold(self, blankThreshold: float,
                                        barcodeSet: pandas.DataFrame
                                        ) -> pandas.DataFrame:
        selectData = barcodeSet[
            ['mean_intensity', 'min_distance', 'area']].values
        selectData[:, 0] = np.log10(selectData[:, 0])
        blankFractionHistogram = self.get_blank_fraction_histogram()

        barcodeBins = np.array(
            (np.digitize(selectData[:, 0], self.get_intensity_bins(),
                         right=True),
             np.digitize(selectData[:, 1], self.get_distance_bins(),
                         right=True),
             np.digitize(selectData[:, 2], self.get_area_bins()))) - 1
        barcodeBins[0, :] = np.clip(
            barcodeBins[0, :], 0, blankFractionHistogram.shape[0]-1)
        barcodeBins[1, :] = np.clip(
            barcodeBins[1, :], 0, blankFractionHistogram.shape[1]-1)
        barcodeBins[2, :] = np.clip(
            barcodeBins[2, :], 0, blankFractionHistogram.shape[2]-1)
        raveledIndexes = np.ravel_multi_index(
            barcodeBins[:, :], blankFractionHistogram.shape)

        thresholdedBlankFraction = blankFractionHistogram < blankThreshold
        return barcodeSet[np.take(thresholdedBlankFraction, raveledIndexes)]

    @staticmethod
    def _extract_counts(barcodes, intensityBins, distanceBins, areaBins):
        barcodeData = barcodes[
            ['mean_intensity', 'min_distance', 'area']].values
        barcodeData[:, 0] = np.log10(barcodeData[:, 0])
        return np.histogramdd(
            barcodeData, bins=(intensityBins, distanceBins, areaBins))[0]

    def _run_analysis(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        barcodeDB = decodeTask.get_barcode_database()

        completeFragments = \
            self.dataSet.load_numpy_analysis_result_if_available(
                'complete_fragments', self, [False]*self.fragment_count())
        pendingFragments = [
            decodeTask.is_complete(i) and not completeFragments[i]
            for i in range(self.fragment_count())]

        areaBins = self.dataSet.load_numpy_analysis_result_if_available(
            'area_bins', self, np.arange(1, 35))
        distanceBins = self.dataSet.load_numpy_analysis_result_if_available(
            'distance_bins', self,
            np.arange(
                0, decodeTask.parameters['distance_threshold']+0.02, 0.01))
        intensityBins = self.dataSet.load_numpy_analysis_result_if_available(
            'intensity_bins', self, None)

        blankCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'blank_counts', self, None)
        codingCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'coding_counts', self, None)

        self.dataSet.save_numpy_analysis_result(
            areaBins, 'area_bins', self)
        self.dataSet.save_numpy_analysis_result(
            distanceBins, 'distance_bins', self)

        updated = False
        while not all(completeFragments):
            if (intensityBins is None or
                    blankCounts is None or codingCounts is None):
                for i in range(self.fragment_count()):
                    if not pendingFragments[i] and decodeTask.is_complete(i):
                        pendingFragments[i] = decodeTask.is_complete(i)

                if np.sum(pendingFragments) >= min(20, self.fragment_count()):
                    def extreme_values(inputData: pandas.Series):
                        return inputData.min(), inputData.max()
                    sampledFragments = np.random.choice(
                            [i for i, p in enumerate(pendingFragments) if p],
                            size=20)
                    intensityExtremes = [
                        extreme_values(barcodeDB.get_barcodes(
                            i, columnList=['mean_intensity'])['mean_intensity'])
                        for i in sampledFragments]
                    maxIntensity = np.log10(
                            np.max([x[1] for x in intensityExtremes]))
                    intensityBins = np.arange(0, 2 * maxIntensity,
                                              maxIntensity / 100)
                    self.dataSet.save_numpy_analysis_result(
                        intensityBins, 'intensity_bins', self)

                    blankCounts = np.zeros((len(intensityBins)-1,
                                            len(distanceBins)-1,
                                            len(areaBins)-1))
                    codingCounts = np.zeros((len(intensityBins)-1,
                                            len(distanceBins)-1,
                                            len(areaBins)-1))

            else:
                for i in range(self.fragment_count()):
                    if not completeFragments[i] and decodeTask.is_complete(i):
                        barcodes = barcodeDB.get_barcodes(
                            i, columnList=['barcode_id', 'mean_intensity',
                                           'min_distance', 'area'])
                        blankCounts += self._extract_counts(
                            barcodes[barcodes['barcode_id'].isin(
                                codebook.get_blank_indexes())],
                            intensityBins, distanceBins, areaBins)
                        codingCounts += self._extract_counts(
                            barcodes[barcodes['barcode_id'].isin(
                                codebook.get_coding_indexes())],
                            intensityBins, distanceBins, areaBins)
                        updated = True
                        completeFragments[i] = True

                if updated:
                    self.dataSet.save_numpy_analysis_result(
                        completeFragments, 'complete_fragments', self)
                    self.dataSet.save_numpy_analysis_result(
                        blankCounts, 'blank_counts', self)
                    self.dataSet.save_numpy_analysis_result(
                        codingCounts, 'coding_counts', self)

class AdaptiveFilterBarcodes(AbstractFilterBarcodes): 

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['adaptive_task'],
                self.parameters['decode_task']]

    def get_adaptive_thresholds(self):
        """ Get the adaptive thresholds used for filtering barcodes.

        Returns: The GenerateaAdaptiveThershold task using for this
            adaptive filter.
        """
        return self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])

    def _run_analysis(self, fragmentIndex):
        adaptiveTask = self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])

        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            self.parameters['misidentification_rate'])

        bcDatabase = self.get_barcode_database()
        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)

        bcDatabase.write_barcodes(adaptiveTask.extract_barcodes_with_threshold(
            threshold, currentBarcodes), fov=fragmentIndex)






class FilterDuplicates(analysistask.AnalysisTask):

    #this function will just read in the hdf files all at once, and then replace what is in the folder 

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

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
        filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task']) 
        bcDatabase = filterTask.get_barcode_database()

        #reading in the barcodes file
        transcript_file = filterTask.get_barcode_database().get_barcodes() #this function is taken from exportbarcodes.py and used in to load all of the data from the adaptive filter task 
        #it also excludes the column list because it just wants all of them and then it will filter them and write them out 
        transcript_file=transcript_file.reset_index(drop=True) 
        #print(transcript_file.head)
        #print(type(transcript_file))

        #reading in the position list 
        position_file= self.dataSet.get_stage_positions()
        position_file=position_file.rename(columns={0: "x", 1: "y"}) 

        # Store indices to remove later
        indices_to_remove = []
        ## REMOVE 2PIXELS FROM THE EDGE
        edge_pixels = 0
        valid_transcripts= pd.DataFrame()
        for i in range(2): #position_file.shape[0]
        #     print(i)
            fov_remove_edges = box(position_file.iloc[i, 0] + edge_pixels, position_file.iloc[i, 1]+edge_pixels, position_file.iloc[i, 0] + 220 - 2*edge_pixels, position_file.iloc[i, 1] +220 - 2*edge_pixels).bounds
        #     print(fov_remove_edges)
            #want to keep these transcripts
            valid_transcripts= pd.concat([valid_transcripts, transcript_file[
                        (transcript_file['fov']==i)&
                        (transcript_file['global_x'] >= fov_remove_edges[0]) & 
                        (transcript_file['global_x'] <= fov_remove_edges[2]) & 
                        (transcript_file['global_y'] >= fov_remove_edges[1]) & 
                        (transcript_file['global_y'] <= fov_remove_edges[3])
                    ]])
# mask = ~transcript_file.index.isin(indices_to_remove)
# print("here")
# output = transcript_file[mask]
        output = valid_transcripts
        # for i in range(position_file.shape[0]):
        #     for j in range(i):
        #         square1_top_left = (position_file.iloc[i, 0], position_file.iloc[i, 1])
        #         square2_top_left = (position_file.iloc[j, 0], position_file.iloc[j, 1])
        #         overlap_region = self.calculate_overlap(square1_top_left, square2_top_left)
        #         if overlap_region[0] is not None:
                    
        #             fov_i_transcripts=transcript_file[
        #                 (transcript_file['fov']==i)&
        #                 (transcript_file['global_x'] >= overlap_region[0]) & 
        #                 (transcript_file['global_x'] <= overlap_region[2]) & 
        #                 (transcript_file['global_y'] >= overlap_region[1]) & 
        #                 (transcript_file['global_y'] <= overlap_region[3])
        #             ].index

        #             fov_j_transcripts=transcript_file[
        #                 (transcript_file['fov']==j)&
        #                 (transcript_file['global_x'] >= overlap_region[0]) & 
        #                 (transcript_file['global_x'] <= overlap_region[2]) & 
        #                 (transcript_file['global_y'] >= overlap_region[1]) & 
        #                 (transcript_file['global_y'] <= overlap_region[3])
        #             ].index

        #             indices_to_remove.extend(fov_j_transcripts)
                         
        # removedtranscripts=transcript_file[transcript_file.index.isin(indices_to_remove)]
        # removedtranscriptspath=self.dataSet.analysisPath+str('/')+self.parameters['filter_task']+str('/barcodes/removedtranscripts')
        # removedtranscripts.to_csv(removedtranscriptspath)

        # mask = ~transcript_file.index.isin(indices_to_remove)
        # output = transcript_file[mask] #output contains the edited transcript_file with all of the potential duplicates removed

        #instead of directly writing output, our goal is to write output one fov at a time to replace the current output files in the AdaptiveFilterBarcodes+ file 
        
        number_of_fovs=position_file.shape[0]
        print("num FOVs", number_of_fovs)
        #print(number_of_fovs)
        #print(self.dataSet.analysisPath+str('/AdaptiveFilterBarcodes/barcodes'))
        #print(self.dataSet.analysisPath+str('/')+self.parameters['filter_task']+str('/barcodes')) #will need to change the str('/barcodes') in case they use a different output folder name, right now it is hard coded 

        for i in range(2): #i is the current fov -- hard coded 51 instead of number_of_fovs
            #create a new pandas dataframe that has the transcripts only from the current fov and then write it to the AdaptiveFilterBarcodes barcodes folder 
            output_for_current_fov=output[(output['fov']==i)]
            #print(i)
            #print(output_for_current_fov.shape)
            #self.dataSet.delete_table(resultName='barcode_data',analysisTask=filterTask,resultIndex=i) #this function not deleting the hdf5 files correctly 
            #delete the hdf5 files that were originally written out by the AdaptiveFilterBarcodes task 
            os.remove(self.dataSet.analysisPath+str('/')+self.parameters['filter_task']+str('/barcodes/barcode_data_')+str(i)+str('.h5')) #this line will work properly as long as the naming scheme of barcodes/barcode_data_fov.h5 is kept, the path to the actual analysis folder and filter task is not hard coded 
            #print(self.dataSet.analysisPath+str('/')+self.parameters['filter_task']+str('/barcodes/barcode_data_')+str(i)+str('.h5'))
            bcDatabase.write_barcodes(output_for_current_fov,fov=i) #write the new hdf file for this fov 

        #bcDatabase.write_barcodes(output) #this is the line to store the entire output 







        '''

class AdaptiveFilterBarcodes(AbstractFilterBarcodes): #this is the old class file that won't work and we need to turn into a new class 

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['adaptive_task'],
                self.parameters['decode_task']]

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

    def get_adaptive_thresholds(self):
        """ Get the adaptive thresholds used for filtering barcodes.

        Returns: The GenerateaAdaptiveThershold task using for this
            adaptive filter.
        """
        return self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])

    def _run_analysis(self, fragmentIndex):
        adaptiveTask = self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            self.parameters['misidentification_rate'])

        #reading in the barcodes file 
        bcDatabase = self.get_barcode_database()
        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)
        transcript_file=adaptiveTask.extract_barcodes_with_threshold(
            threshold, currentBarcodes)
        transcript_file=transcript_file.reset_index(drop=True) #check to see if resetting the indices affects merlin downstream 
        #does this transcript file have different information than the algorithm is meant to handle? 

        pandas.set_option('display.max_columns', None)
        #print(transcript_file.head)

        #reading in the position list 
        position_file= self.dataSet.get_stage_positions()
        position_file=position_file.rename(columns={0: "x", 1: "y"}) 

        print(position_file)
        print(transcript_file['fov'].unique())


        # Store indices to remove later
        indices_to_remove = []
        for i in range(position_file.shape[0]):
            for j in range(i):
                square1_top_left = (position_file.iloc[i, 0], position_file.iloc[i, 1])
                #print(square1_top_left)
                square2_top_left = (position_file.iloc[j, 0], position_file.iloc[j, 1])
                overlap_region = self.calculate_overlap(square1_top_left, square2_top_left)
                if overlap_region[0] is not None:
                    
                    fov_i_transcripts=transcript_file[
                        (transcript_file['fov']==i)
                        (transcript_file['global_x'] >= overlap_region[0]) & 
                        (transcript_file['global_x'] <= overlap_region[2]) & 
                        (transcript_file['global_y'] >= overlap_region[1]) & 
                        (transcript_file['global_y'] <= overlap_region[3])
                    ].index

                    fov_j_transcripts=transcript_file[
                        (transcript_file['fov']==j)
                        (transcript_file['global_x'] >= overlap_region[0]) & 
                        (transcript_file['global_x'] <= overlap_region[2]) & 
                        (transcript_file['global_y'] >= overlap_region[1]) & 
                        (transcript_file['global_y'] <= overlap_region[3])
                    ].index

                    number_of_transcripts_in_fov_i=len(fov_i_transcripts)
                    number_of_transcripts_in_fov_j=len(fov_j_transcripts)
                    #print("i is",i)
                    #print("j is",j)
                    #print(number_of_transcripts_in_fov_i)
                    #print(number_of_transcripts_in_fov_j)
                    if number_of_transcripts_in_fov_i < number_of_transcripts_in_fov_j:
                        #print("i is less")
                        indices_to_remove.extend(fov_i_transcripts)
                    elif number_of_transcripts_in_fov_j < number_of_transcripts_in_fov_i:
                        #print("j is less")
                        indices_to_remove.extend(fov_j_transcripts)

        mask = ~transcript_file.index.isin(indices_to_remove)
        output = transcript_file[mask] #output contains the edited transcript_file with all of the potential duplicates removed
        #print(len(indices_to_remove))
        #print(output.shape)
        bcDatabase.write_barcodes(output, fov=fragmentIndex)

        '''
