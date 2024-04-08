from merlin.core import analysistask
import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Polygon, box
from rtree import index as rindex
import numpy as np
from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import _20230908_spatialfeature as spatialfeature
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

class VolumeCellMetadata(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        
        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task']]
    
    def _run_analysis(self, fragmentIndex) -> None:
        featureDB = self.get_feature_database()
        features = self.segmentTask.get_feature_database().read_features(fragmentIndex)
        featureDB.write_features(features, fragmentIndex)

class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.volumeTask = self.dataSet.load_analysis_task(
            self.parameters['volume_recalculations'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['volume_recalculations']]

    def _run_analysis(self):
        df = self.volumeTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata_merlin2',
                                           self.analysisName)