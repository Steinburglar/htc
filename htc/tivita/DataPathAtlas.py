"""
Lucas Steinberger
04.06.2024
draft of Child DataPath class module, to be used to make the existing htc framework compatible with the tissue atlas folder structure.
Main goals:
    -revise iterate function so that it matches structure of tissue atlas. (starting from "data" directory in larger dataset. )
    -retain each paths ability to find its subject and other data, that is useful for filters. 
    -deal with dependencies in the dataPath object that expect a "data" and "intermediates" directory
    -enable user to externally and explicitly specify a dataset_settings.json file for their dataset, so it can live outside the folder structure.
    """


#imports based on DataPathTivita.py
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Union

from typing_extensions import Self

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.tivita.DataPathTivita import DataPathTivita

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
            -iterate(): iterates through all available images and creates a Path to each image. this is the primary way to load a 
                for training/inference DONE!! :)
            -from_image_path(): less flexible that from_image_name, losses some information. 
            

        Args:
            data_dir: The path where the data is stored. The data folder should contain a dataset_settings.json file.
            filters: List of filters which can be used to alter the set of images returned by this function. Every filter receives a DataPath instance and the instance is only yielded when all filter return True for this path.
            annotation_name: Include only paths with this annotation name and use it as default in read_segmentation(). Must either be a lists of annotation names or as string in the form name1&name2 (which will automatically be converted to ['name1', 'name2']). If None, no default annotation name will be set and no images will be filtered by annotation name.
            Dataset_settings: Path to the desired dataset_settings.json folder. This is a parameter specific to the DataPathAtlass class, added to allow a user
            to explicitly define a dataset_settings json that is not inside the dataset, in the case of working with the Tissue atlas Data structure. 
        Returns: Generator with all path objects
    """
    @staticmethod
    def from_image_name(image_name: str) -> Self:
        raise NotImplementedError()
    
    
    
    
    @staticmethod
    def iterate(
        data_dir: Path,
        filters: list[Callable[[Self], bool]],
        annotation_name: Union[str, list[str]],
        dataset_settings: Path,
    ) -> Iterator["DataPathAtlas"]:
        """
        Backbone helper function to load up paths to the data we want. returns a generator/iterator
        Args:
            data_dir: path to "data" subfolder of the dataset
            filters: list of filter funtions that return 
        """
        # Settings of the dataset (shapes etc.) can be referenced by the DataPaths
        intermediates_dir = settings.datasets.find_intermediates_dir(data_dir)
        dataset_settings_dict = {}
        dataset_settings_dict[dataset_settings] = DatasetSettings(dataset_settings) #here the key is the pathr
        #currently only works with explicitly specified location for dataset_settings file. awkward syntax is remnant of feature
        #that allows user to have multiple json,s with some taking precedence. not yet implemented for this class.
        """
        parent_paths = list(data_dir.parents)
        parent_paths.reverse()
        for p in parent_paths:
            if (p / "dataset_settings.json").exists():
                dataset_settings_dict[p] = DatasetSettings(p / "dataset_settings.json")
        """
        # Keep a list of used image folders in case a folder contains both a cube file and a tiv archive
        used_folders = set()
        for subject_dir in data_dir.iterdir():
            for timestamp_dir in subject_dir.iterdir():
                for root, dirs, files in os.walk(timestamp_dir):
                    dirs.sort()  # Recurse in sorted order
                    for f in sorted(files):
                        if f.endswith("SpecCube.dat") and root not in used_folders:
                            if len(dataset_settings_dict) == 0:
                                dataset_settings = None
                            else:
                                dataset_settings = list(dataset_settings_dict.values())[
                                    -1
                                ]  # last dict item should be closest to path
                            path = DataPathAtlas(Path(root), data_dir, intermediates_dir, dataset_settings, annotation_name)
                            if all(f(path) for f in filters):
                                yield path
                            used_folders.add(root)
                            
