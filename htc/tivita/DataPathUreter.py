
# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT
"""
Lucas Steinberger
04.06.2024
Child DataPath class module, to be used on the Urology groups ureter folder structure.
"""
import importlib
import json
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
import pandas as pd
from PIL import Image
from typing_extensions import Self

from htc.settings import settings
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config
from htc.utils.JSONSchemaMeta import JSONSchemaMeta
from htc.utils.LabelMapping import LabelMapping
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathAtlas2 import DataPathAtlas

class DataPathUreter(DataPath):
    def __init__(
        self,
        image_dir: Union[str, Path, None],
        data_dir: Union[str, Path] = None,
        intermediates_dir: Union[str, Path] = None,
        dataset_settings: DatasetSettings = None,
        annotation_name_default: Union[str, list[str]] = None,):
        
        """Constructs a DataPathUreter object that enables the user to employ the htc framework on the ureter data structure.
        designed to be created/Accessed through three static functions:
            -from_image_name(): searches through available data and generates a Path object to a specific image, whose name is given as the arg
            -iterate(): iterates through all available images and creates a Path to each image. this is the primary way to load a 
                for training/inference
            -from_image_path(): less flexible that from_image_name, losses some information. 
            
        Args:
            image_dir: Path (or string) to the image directory (timestamp folder).
            data_dir: Path (or string) to the data directory of the dataset (it should contain a dataset_settings.json file).
            intermediates_dir: Path (or string) to the intermediates directory of the dataset.
            dataset_settings: Reference to the settings of the dataset. If None and no settings could be found in the image directory, the parents of the image directory are searched. If available, the closest dataset_settings.json is used. Otherwise, the data path gets an empty dataset settings assigned.
            annotation_name_default: Default annotation_name(s) which will be used when reading the segmentation with read_segmentation() with no arguments.
        """
        #use super().__init__ to pass arguments to DataPath __init__. This way the DataPathAtlas object has all the same attributes that a DataPath Object would.
        super().__init__(image_dir, data_dir, intermediates_dir, dataset_settings, annotation_name_default)
        self.timepoint_dir = self.image_dir.parent
        self.subject_name = self.timepoint_dir.parent.name #helpful for folding by subject
        self.grandparent_dir = self.timepoint_dir.parent.parent.parent
        self.subdataset_name = self.grandparent_dir.name
    
    def image_name(self) -> str: #most meta.feather files seem to be organized in this way
        name = f"{self.subject_name}#{self.timestamp}"
        #keep in mind the that the base DataPath has a __repr__ method that just returns the string path (not name) of the image dir.
    	#this does NOT overwrite that function.
        return name
    def contains_txt(self):
        """
        Method to check if the timestamp folder of the given path object contains a certain .txt file, which is included in only valid data captures
        Useful for writing a filter function to filter out images missing this .txt file
        Returns:
            Bool: returns True if the file does exist in the timestamp folder, False otherwise
        """
        file_path = self.image_dir / '_labelling_001.txt'
        return file_path.is_file()
        
        ##write the syntax when you can look it up and know what the file looks like
    
    def links_and_rechts(self):
        """
        Check if an image directory indicates the presence of both left and right ureters.
        Returns True if both ureters are present, otherwise returns False.
        Checks by looking at the parent folder of the timestamp folder.
        In the case where right and left are separated, there will be an extra folder in the path labelled "rechts" or "links"
        """
        dir_name = self.image_dir.parent.name
        return not ('links' in dir_name or 'rechts' in dir_name)

    
    
    
    
    def contains_MITK(self):
        """
        Method to check if the MITK folder of the given path object contains any .nrrd file.
        Useful for writing a filter function to filter out images missing .nrrd files in the MITK folder.
        Returns:
            Bool: returns True if at least one .nrrd file exists in the MITK folder, False otherwise
        """
        mitk_folder = self.image_dir / 'MITK'
        
        if not mitk_folder.is_dir():
            return False
        
        nrrd_files = list(mitk_folder.glob('*.nrrd'))
        
        return len(nrrd_files) > 0
        
    @staticmethod
    def iterate(
        data_dir: Path,
        filters: list[Callable[[Self], bool]] = None,
        annotation_name: Union[str, list[str]] = None,
        
    ) -> Iterator["DataPathAtlas"]:
        """
        Backbone helper function to load up paths to the data we want. returns a generator/iterator
        Args:
            data_dir: The path where the data is stored.
            filters: List of filters which can be used to alter the set of images returned by this function. Every filter receives a DataPath instance and the instance is only yielded when all filter return True for this path.
            annotation_name: Include only paths with this annotation name and use it as default in read_segmentation(). Must either be a lists of annotation names or as string in the form name1&name2 (which will automatically be converted to ['name1', 'name2']). If None, no default annotation name will be set and no images will be filtered by annotation name.
        """
        if isinstance(data_dir, list):  # if given a list of data_dir paths
            for data in data_dir:
                yield from DataPathUreter.iterate(data, filters, annotation_name)
            return
        
        assert isinstance(data_dir, Path), "DataPathAtlas.iterate must be given a path to a 'data' folder, or a list of such paths"
        
        
        
        # Settings of the dataset (shapes etc.) can be referenced by the DataPaths
        dataset_settings = None
        if filters == None:
            filters = []
        if settings.external_dir is not None: #this is a datasets object
            ext_exists = True
            entry = settings.external_dir['PATH_HTC_EXTERNAL']
            external_dir = entry["path_dataset"] #this is a path/string, as opposed to a datasets object
            intermediates_dir = entry["path_intermediates"]
        else:
            ext_exists = False
            intermediates_dir = settings.datasets.find_intermediates_dir(data_dir) #remove this line once i have set up external intermediates dir
        
        if ext_exists and (external_dir /'data'/ "dataset_settings.json").exists():
            dataset_settings = DatasetSettings(external_dir /'data'/ "dataset_settings.json")
        elif (data_dir / "dataset_settings.json").exists():
            dataset_settings = DatasetSettings(data_dir / "dataset_settings.json")
        else:
            dataset_settings = None
        # Keep a list of used image folders in case a folder contains both a cube file and a tiv archive
        used_folders = set()
        for subject_dir in data_dir.iterdir():
            for timepoint_dir in subject_dir.iterdir():
                for timestamp_dir in timepoint_dir.iterdir():
                    for root, dirs, files in os.walk(timestamp_dir):
                        dirs.sort()  # Recurse in sorted order
                        for f in sorted(files):
                            if f.endswith("SpecCube.dat",) and root not in used_folders:
                                path = DataPathUreter(Path(root), data_dir, intermediates_dir, dataset_settings, annotation_name)
                                if all(f(path) for f in filters):
                                    used_folders.add(root)
                                    yield path