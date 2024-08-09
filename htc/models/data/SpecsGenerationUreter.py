# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from abc import abstractmethod
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from typing import Callable, Union, Self
from collections import defaultdict
import pandas as pd

from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPathAtlas2 import DataPathAtlas
from htc import SpecsGeneration

class SpecsGenerationUreter(SpecsGeneration):
    
    def __init__(self,
        intermediates_dir: Path,
        filters: list[Callable[[Self], bool]] = None,
        annotation_name: Union[str, list[str]] = None,
        test_ratio: float = 0.2,
        n_folds: int = 5,
        seed: int = 42,
        name = "Atlas"
        ):
        #should be built off of L1, Segmentatioons, or meta table, to avoid duplicates
        #should also fold by subject
        # Unique name of the resulting specs file
        super().__init__(name = name)
        self.intermediates_dir = intermediates_dir
        self.filters = filters
        self.annotation_name = annotation_name
        self.test_ratio = test_ratio
        self.n_folds = n_folds
        self.seed = seed
        
    def group_by_subject(self, df):
        subject_groups = defaultdict(list)
        for _, row in df.iterrows():
            image_name = row['image_name']
            subject_name = image_name.split('#')[0]  # Assuming subject name is the first part of the image name
            subject_groups[subject_name].append(image_name)
        return subject_groups

    
    
    
    
    def generate_folds(self,
        test_ratio: float = 0.2,
        n_folds: int = 5
    ) -> list[dict]: 
        test_ratio = self.test_ratio
        n_folds = self.n_folds
        seed = self.seed
        table_path = list((self.intermediates_dir/ "tables").glob("*@meta.feather"))
        assert len(table_path) > 0, "no meta.feather found in the specified intermediates directory"
        assert len(table_path) == 1, f"More than one meta table found in the specified intermediates directory. Try regenerating the contents of your external directory with rundatasetalex.py, --regenerate"
        table_path = table_path[0]
        df = pd.read_feather(table_path)
        subject_groups = self.group_by_subject(df)
        subjects = list(subject_groups.keys())
        print(subjects)       
        if test_ratio == 0: #no testing data
            train_subjects, test_subjects = subjects, []
        else:
            train_subjects, test_subjects = train_test_split(subjects, test_size=self.test_ratio, random_state=self.seed)
        test_images = [img for sub in test_subjects for img in subject_groups[sub]] #creates list of image names in the test set

        kf = KFold(n_splits=n_folds, shuffle=True, random_state = self.seed)
        split = 1
        data_specs = []
        for train_indices,val_indices,  in kf.split(train_subjects):
            #iterate through the folds, creating a fold_specs dictionary detailing each fold
            train_subs = [train_subjects[i] for i in train_indices]
            val_subs = [train_subjects[i] for i in val_indices]
            train_images = [img for sub in train_subs for img in subject_groups[sub]]
            val_images = [img for sub in val_subs for img in subject_groups[sub]]
            
            fold_specs = {
                "fold_name": f"fold_{split}",
                "train": {
                    "image_names": train_images,
                    "data_path_module": "htc.tivita.DataPathUreter",
                    "data_path_class": "DataPathUreter",
                },
                "val": {
                    "image_names": val_images,
                    "data_path_module": "htc.tivita.DataPathUreter",
                    "data_path_class": "DataPathUreter",
                },
                "test": {
                    "image_names": test_images,
                    "data_path_module": "htc.tivita.DataPathUreter",
                    "data_path_class": "DataPathUreter",
                },
            }
            data_specs.append(fold_specs)
            split += 1
            
        return data_specs #list of dictionaries, each of which specifies a test/validation fold