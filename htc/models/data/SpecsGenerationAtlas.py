# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from abc import abstractmethod
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from typing import Callable, Union, Self

from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPathAtlas import DataPathAtlas
from htc import SpecsGeneration

class SpecsGenerationAtlas(SpecsGeneration):
    
    def __init__(self,
        data_dir: Path,
        dataset_settings: Path,
        filters: list[Callable[[Self], bool]] = None,
        annotation_name: Union[str, list[str]] = None,
        test_ratio: float = 0.2,
        n_folds: int = 5,
        seed: int = 42,
        name = "Atlas"):
        # Unique name of the resulting specs file
        super().__init__(name = name)
        self.data_dir = data_dir
        self.dataset_settings = dataset_settings
        self.filters = filters
        self.annotation_name = annotation_name
        self.test_ratio = test_ratio
        self.n_folds = n_folds
        self.seed = seed
        
    def generate_folds(self,
        test_ratio: float = 0.2,
        n_folds: int = 5,
        seed: int = 42
    ) -> list[dict]: 
        test_ratio = self.test_ratio
        n_folds = self.n_folds
        seed = self.seed
        paths = list(DataPathAtlas.iterate(self.data_dir, dataset_settings = self.dataset_settings, filters = self.filters, annotation_name= self.annotation_name)) #add intermediates_dir when implemented
        if test_ratio == 0:
            paths_fold, paths_test = paths, []
        else:
            paths_fold, paths_test = train_test_split(paths, test_size = test_ratio, )
        imgs_test = [p.image_name() for p in paths_test] #creates list of image names in the test set

        fold = KFold(n_splits=n_folds, shuffle=True, random_state = seed)
        split = 1
        data_specs = []
        for val_indices, train_indices in fold.split(paths_fold):
            #iterate through the folds, creating a fold_specs dictionary detailing each fold
            paths_val = [paths_fold[i] for i in val_indices]
            paths_train = [paths_fold[i] for i in train_indices]
            imgs_val = [p.image_name() for p in paths_val]
            imgs_train = [p.image_name() for p in paths_train]
            
            fold_specs = {
                "fold_name": f"fold_{split}",
                    
                "train": {
                    "image_names": imgs_train,
                    "data_path_module": "htc.tivita.DataPathAtlas",
                    "data_path_class": "DataPathAtlas",
                    "dataset_settings": str(self.dataset_settings)
                },
                "val": {
                    "image_names": imgs_val,
                    "data_path_module": "htc.tivita.DataPathAtlas",
                    "data_path_class": "DataPathAtlas",
                    "dataset_settings": str(self.dataset_settings)
                },
                "test": {
                    "image_names": imgs_test,
                    "data_path_module": "htc.tivita.DataPathAtlas",
                    "data_path_class": "DataPathAtlas",
                    "dataset_settings": str(self.dataset_settings)
                },
            }
            data_specs.append(fold_specs)
            split +=1
            
        return data_specs #list of dictionaries, each of which specifies a test/validation fold