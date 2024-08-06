# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT
"""
Lucas Steinberger
04.06.2024
NOW DEFUNCT. USE  DataPathAtlas2.py


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

#imports of DataPath
import importlib
import json
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

class DataPathAtlas(DataPath):
    def __init__(
        self,
        image_dir: Union[str, Path, None],
        data_dir: Union[str, Path] = None,
        intermediates_dir: Union[str, Path] = None,
        dataset_settings: DatasetSettings = None,
        annotation_name_default: Union[str, list[str]] = None,):
        
        """Constructs a DataPathAtlas object that enables the user to employ the htc framework on the dkfz tissue atlas data structure.
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
        self.subject_name = self.image_dir.parent.name #helpful for folding by subject
    
    def image_name(self) -> str: #most meta.feather files seem to be organized in this way
        name = f"{self.subject_name}#{self.timestamp}"

        return name
    
    @staticmethod
    def _build_cache(dataset_settings: Path, local=True) -> dict[str, Any]:  #modified version to allow for json path specification
        # We use a dict for the cache because it is much faster than a dataframe
        cache = {}
        print("_build_cache was acessed")
        for env_key in settings.datasets.env_keys():
            if not env_key.upper().startswith("PATH_TIVITA"):
                continue

            entry = settings.datasets.get(env_key, local_only=local)
            if entry is None:
                continue

            if (local and entry["location"] == "local") or (not local and entry["location"] == "network"):
                table_path = list((entry["path_intermediates"] / "tables").glob("*@meta.feather"))
                if len(table_path) > 0:
                    assert len(table_path) == 1, f"More than one meta table found for {entry}"
                    table_path = table_path[0]

                    
                    dsettings = DatasetSettings(dataset_settings)
                    df = pd.read_feather(table_path)
                    df["dsettings"] = dsettings
                    df["dataset_env_name"] = env_key
                    df["data_dir"] = entry["path_data"]
                    df["intermediates_dir"] = entry["path_intermediates"]

                    # Append the metadata for the current dataset to the global cache
                    df.set_index("image_name", inplace=True)
                    cache |= df.to_dict("index")

        return cache
    
    @staticmethod
    def _local_cache(dataset_settings:Path,) -> dict[str, Any]:
        if DataPathAtlas._local_meta_cache is None:
            DataPathAtlas._local_meta_cache = DataPathAtlas._build_cache(dataset_settings, local= True)

        return DataPathAtlas._local_meta_cache

    @staticmethod
    def _network_cache(dataset_settings: Path) -> dict[str, Any]:
        if DataPathAtlas._network_meta_cache is None:
            DataPathAtlas._network_meta_cache = DataPathAtlas._build_cache(dataset_settings, local=False)

        return DataPathAtlas._network_meta_cache

    @staticmethod
    def _find_image(image_name: str, dataset_settings: Path) -> dict[str, Any]:
        cache = DataPathAtlas._local_cache(dataset_settings)
        if image_name not in cache:
            # Avoid building the network cache if possible
            cache = DataPathAtlas._network_cache(dataset_settings)
            if image_name not in cache:
                return None

        return cache[image_name]

    @staticmethod
    def image_name_exists(image_name: str, dataset_settings: Path) -> bool:
        """
        Checks whether the image name can be found.

        >>> DataPathAtlas.image_name_exists('P043#2019_12_20_12_38_35')
        True

        Args:
            image_name: Unique identifier of the path in the same format as for `from_image_name()`.

        Returns: True if the image name can be found, False otherwise.
        """
        if "@" in image_name:
            image_name = image_name.split("@")[0]

        if image_name.startswith("ref"):
            from htc.tivita.DataPathReference import DataPathReference

            return DataPathReference.image_name_exists(image_name)
        else:
            match = DataPathAtlas._find_image(image_name, dataset_settings)
            return match is not None

    @staticmethod
    def from_image_name(image_name: str, dataset_settings: Path) -> Self:
        """
        Constructs a data path based on its unique identifier.

        This function can only be used if the corresponding dataset has a *meta.feather table with an overview of all paths of the dataset. This table can be created via the run_meta_table script.

        Note: Data path objects created via this method are cached and exists globally during program execution. If the same `image_name` is used again, then the reference to the same object as before is returned.

        Args:
            image_name: Unique identifier of the path. Usually in the form subject#timestamp but you can also extend it to define the default annotations to read, for example subject#timestamp@name1&name2.

        Returns: The data path object.
        """
        # The same path but with different annotation names should also be separate path objects since each object has its own annotation_name_default
        cache_name = image_name

        if image_name not in DataPathAtlas._data_paths_cache:
            if "@" in image_name:
                image_name, annotation_name = image_name.split("@")
                annotation_name = annotation_name.split("&")
                if len(annotation_name) == 1:
                    annotation_name = annotation_name[0]
            else:
                annotation_name = None

            if image_name.startswith("ref"):
                from htc.tivita.DataPathReference import DataPathReference

                DataPathAtlas._data_paths_cache[cache_name] = DataPathReference.from_image_name(image_name, annotation_name)
            else:
                match = DataPathAtlas._find_image(image_name, dataset_settings)
                assert match is not None, (
                    f"Could not find the path for the image {image_name} ({len(DataPathAtlas._local_cache()) = },"
                    f" {len(DataPathAtlas._network_cache()) = })"
                )
                DataPathClass = DataPathAtlas #had to just hard code it in, otherwise it would require altering the whole framework. 
                if DataPathClass is None:
                    raise ValueError(
                        f"No known DataPath class for the dataset {match['dataset_env_name']}. Please make sure that"
                        " you have a dataset_settings.json file in your dataset which has a key data_path_class which"
                        " refers to a valid data path class (e.g. htc.tivita.DataPathMultiorgan>DataPathMultiorgan)"
                    )

                DataPathAtlas._data_paths_cache[cache_name] = DataPathClass(
                    match["data_dir"] / match["path"],
                    match["data_dir"],
                    match["intermediates_dir"],
                    match["dsettings"],
                    annotation_name,
                )

        return DataPathAtlas._data_paths_cache[cache_name]
    
    
    @staticmethod
    def iterate(
        data_dir: Path,
        dataset_settings: Path,
        #intermediates_dir: Path,
        filters: list[Callable[[Self], bool]] = None,
        annotation_name: Union[str, list[str]] = None,
        
    ) -> Iterator["DataPathAtlas"]:
        """
        Backbone helper function to load up paths to the data we want. returns a generator/iterator
        Args:
            data_dir: path to "data" subfolder of the dataset
            dataset_settings: path to dataset_settings.json
            intermediates_dir: path to external intermediates_dir
            filters: list of filter functions that return booleans. called on the paths
        """
        # Settings of the dataset (shapes etc.) can be referenced by the DataPaths
        if filters == None:
            filters = []
        intermediates_dir = settings.datasets.find_intermediates_dir(data_dir) #remove this line once i have set up external intermediates dir
        dataset_settings_dict = {}
        dataset_settings_dict[dataset_settings] = DatasetSettings(dataset_settings) #here the key is the path
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
                                print("dataset_sttings was set to none")
                            else:
                                dataset_settings = list(dataset_settings_dict.values())[
                                    -1
                                ]  # last dict item should be closest to path
                            path = DataPathAtlas(Path(root), data_dir, intermediates_dir, dataset_settings, annotation_name)
                            if all(f(path) for f in filters):
                                used_folders.add(root)
                                yield path
                            
                            
