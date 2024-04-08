import pandas
import numpy as np

from merlin.core import analysistask
from merlin.util import spatialfeature
import datetime

class PartitionBarcodes(analysistask.ParallelAnalysisTask):

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
                self.parameters['alignment_task']]
    
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
            concatenated_df = pandas.concat(
                [self.get_partitioned_barcodes(fov)
                 for fov in self.dataSet.get_fovs()])
            
            return concatenated_df.groupby(concatenated_df.index).sum()

        return self.dataSet.load_dataframe_from_csv(
            'counts_per_cell', self.get_analysis_name(), fov, index_col=0)
    
#    def _reset_analysis(self, fragmentIndex: int = None) -> None:
#         super()._reset_analysis(fragmentIndex)
#         self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)
    
    # def get_partition_df(self, fov: int = None) -> pandas.DataFrame:
    #     if(self.partition_df == None):
    #         self.partition_df = pandas.DataFrame()

    #     return self.partition_df
    
    def _run_analysis(self, fragmentIndex):
        print("parition begin")
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['alignment_task'])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()
        currentFOVBarcodes = bcDB.get_barcodes(fragmentIndex, 
            columnList=["barcode_id", "global_x", "global_y", "z"])

        # for fi in fovIntersections:
        #     partialBC = bcDB.get_barcodes(fi)
        #     if fi == fovIntersections[0]:
        #         currentFOVBarcodes = partialBC.copy(deep=True)
        #     else:
        #         currentFOVBarcodes = pandas.concat(
        #             [currentFOVBarcodes, partialBC])

        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        countsDF = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells])

        cells_w_transcripts = []

        for cell in currentCells:
            contained = cell.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
            count = currentFOVBarcodes[contained].groupby('barcode_id').size()
            count = count.reindex(range(barcodeCount), fill_value=0)
            countsDF.loc[cell.get_feature_id(), :] = count.values.tolist()
            cell.set_transcript_list(currentFOVBarcodes[contained])
            # print(currentFOVBarcodes[contained])
            cells_w_transcripts.append(cell)

            # print(cell.get_transcript_list()
        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)
        
        self.get_feature_database().write_feature_transcripts(cells_w_transcripts, fragmentIndex)
        print("parition finish")
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
        # print(parsedBarcodes.index)
        self.dataSet.save_dataframe_to_csv(
                    parsedBarcodes, 'barcodes_per_feature',
                    self.get_analysis_name())
        
        # export hdf5 information
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        
        # for fov in self.dataSet.get_fovs():
        #     sDB = pTask.get_feature_database()
        #     currentCells = sDB.read_features(fov)
        #     with sDB._dataSet.open_hdf5_file(
        #             'a', 'feature_data', "ExportPartitionedBarcodes", fov, 'features') \
        #             as f:
        #         featureGroup = f.require_group('featuredata')
        #         # featureGroup.attrs['version'] = merlin.version()
        #         for cell in currentCells:
        #             sDB._save_feature_to_hdf5_group(featureGroup,
        #                                                 cell,
        #                                                 fov)
        #             print(cell.get_transcript_list())
        # parsedBarcodes = pTask.get_partitioned_barcodes(0)
        # # print(parsedBarcodes.index)
        # self.dataSet.save_dataframe_to_csv(
        #             parsedBarcodes, 'fov0_barcodes_per_feature',
        #             self.get_analysis_name())
        # parsedBarcodes = pTask.get_partitioned_barcodes(1)

        # self.dataSet.save_dataframe_to_csv(
        #             parsedBarcodes, 'fov1_barcodes_per_feature',
        #             self.get_analysis_name())