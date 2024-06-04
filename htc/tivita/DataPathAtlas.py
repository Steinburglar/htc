"""
Lucas Steinberger
04.06.2024
draft of Child DataPath class module, to be used to make the existing htc framework compatible with the tissue atlas folder structure.
Main goals:
    -revise iterate function so that it knows how to path own variable lengths to the image folders.
    -retain each paths ability to find its subject and other data, that is useful for filters. 
    -deal with dependencies in the dataPath object that expect a "data" and "intermediates" directory
    -enable user to externally and explicitly specify a dataset_settings.json file for their dataset, so it can live outside the folder structure.
    """


#imports based on DataPathMultiorgan.py
import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Union

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings

class DataPathAtlas(DataPath):
    def __init__(
        self,
        image_dir: Union[str, Path, None],
        data_dir: Union[str, Path] = None,
        intermediates_dir: Union[str, Path] = None,
        dataset_settings: DatasetSettings = None,
        annotation_name_default: Union[str, list[str]] = None,):
        #use super().__init__ to pass arguments to DataPath __init__. This way the DataPathAtlas object has all the same attributes that a DataPath Object would.
        super().__init__(image_dir, data_dir, intermediates_dir, dataset_settings, annotation_name_default)
    """Constructs a DataPathAtlas object that enables the user to employ the htc framework on the dkfz tissue atlas data structure.
        designed to be created/Accessed through three static functions:
            -from_image_name(): searches through available data and generates a Path object to a specific image, whose name is given as the arg
            -iterate(): iterates through all available images and creates a Path to each image. this is the primary way to load a dataset
                for training/inference
            -from_image_path(): less flexible that from_image_name, losses some information. 
            

    Args:
        image_dir: Path (or string) to the image directory (timestamp folder).
        data_dir: Path (or string) to the data directory of the dataset (it should contain a dataset_settings.json file).
        intermediates_dir: Path (or string) to the intermediates directory of the dataset.
        dataset_settings: Reference to the settings of the dataset. If None and no settings could be found in the image directory, the parents of the image directory are searched. If available, the closest dataset_settings.json is used. Otherwise, the data path gets an empty dataset settings assigned.
        annotation_name_default: Default annotation_name(s) which will be used when reading the segmentation with read_segmentation() with no arguments.
        
    """
    