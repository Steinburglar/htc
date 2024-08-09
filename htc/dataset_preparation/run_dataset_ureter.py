import argparse
import copy
import itertools
import json
import re
import shutil
from functools import cached_property, partial
from urllib.parse import quote_plus
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Union
import numpy as np
import pandas as pd
from PIL import Image
from rich.progress import track
from typing_extensions import Self, Generator
from htc.tivita.DataPathUreter import DataPathUreter
from htc.data_processing.run_l1_normalization import L1Normalization
from htc.data_processing.run_median_spectra import MedianSpectra
from htc.data_processing.run_parameter_images import ParameterImages
from htc.data_processing.run_raw16 import Raw16
#from htc.data_processing.run_rgb_sensor_aligned import RGBSensorAligned
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.dataset_preparation.run_dataset_urology import DatasetGeneratorUrology
from htc.tivita.DatasetSettings import DatasetSettings
from htc.tivita.metadata import generate_metadata_table
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.general import clear_directory, safe_copy
from htc.utils.mitk.mitk_masks import nrrd_mask
from htc.utils.parallel import p_map
from htc.utils.paths import filter_min_labels
from htc.utils.LabelMapping import LabelMapping
from htc.utils.blosc_compression import compress_file

from htc.utils.helper_functions import sort_labels
from htc.utils.visualization import compress_html, create_overview_document


class DatasetGeneratorUreter(DatasetGeneratorUrology):
    def __init__(self, **kwargs):
    
        """
        Class to process data from ureter dataset. identical to DatasetGeneratorUrology, except that the dataset_paths method uses DataPathUreter, as opposed to DataPathAtlas,
        to deal with the unique file structure of the ureters. unforntunateöy, we cannot just use DataPath for both, because it would need a dataset_settings to know which DataPath class to use, and
        the datset_settings is generated DURING this script, not before.

        Args:
            output_path: Path where the intermediates and dataset_settings should be stored (data and intermediates subfolder will be created at this location).
            input_path: LIST of Paths to the original dataset from the clinicians (e.g. folders with the name `Cat_*`).
            paths: Explicit list of paths which should be considered in the methods. If None, all found images in the output path will be used (usually what you want).
            regenerate: If True, will delete existing intermediate files before generating new ones.
            include_csv: If True, will save tables additionally as CSV files.
        """
        super().__init__(**kwargs)


    @property
    def dataset_paths(self) -> dict[str, list[DataPath]]:
        """
        Dictionary with dataset name as key and list of paths as value. 
        ```
        designed to allow for using multiple "subdatasets" at once (i.e, multiple different directories
        in the original atlas structure that contain the directory "data")
        
        Important to note that the DataPath class already has the functionality to directly recieve a list of datapaths. However, it only generates one large list.
        furthermore, the subdirectories method of this class can iterate through a liost of all datapaths and reconstruct a dictionary sorting them by subdataset.
        Therefore, the functionaly of this method can be recontructed, so you have multiple options.
        """
        _dict = {}
        data_subdirectories = [path / 'data' for path in self.input_path]
        print(data_subdirectories)
        filter_rl = lambda p: p.links_and_rechts()
        filter_mitk = lambda p: p.contains_MITK()
        filters = [filter_rl]
        if self.copied_mitk:
            filters.append(filter_mitk)
            print("now filtering for paths woth MITK annotations")
        for data_dir in data_subdirectories:
            paths = list(DataPathUreter.iterate(data_dir, filters=filters))
            print(len(paths))
            name0 = paths[0].subdataset_name
            assert all(path.subdataset_name == name0 for path in paths), "Not all items are from the same subdataset"
            _dict[name0] = paths #set the ist of paths to
            
        return _dict

    def dataset_settings(self) -> None:
        label_mapping, last_valid_label_index = self._hypergui_mitk_label_mapping()

        dataset_settings = {
            "dataset_name": "2021_02_05_Tivita_multiorgan_masks",
            "data_path_class": "htc.tivita.DataPathUreter>DataPathUreter",
            "shape": [480, 640, 100],
            "shape_names": ["height", "width", "channels"],
            "label_mapping": label_mapping,
            "last_valid_label_index": last_valid_label_index,
            "annotation_name_default": "polygon#annotator1",
            "annotator_mapping": {
                "annotator1": "Marc Bressan",
                "annotator2": "Alexander Studier-Fischer",
                "annotator3": "Berkin Özdemir",
                "annotator4": "Sarah Bernhardt",
                "annotator5": "Silvia Seidlitz",
            },
        }

        with (self.data_dir / "dataset_settings.json").open("w") as f:
            json.dump(dataset_settings, f, cls=AdvancedJSONEncoder, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # The path argument for DatasetGenerator can be set to 2021_02_05_Tivita_multiorgan_masks dataset paths. Example:
    # screen -S dataset_masks -d -m script -q -c "htc dataset_masks --input-path /mnt/E130-Projekte/Biophotonics/Data/2020_07_23_hyperspectral_MIC_organ_database/data/Catalogization_tissue_atlas" dataset_masks.log
    external_path = external = settings.external_dir['PATH_HTC_EXTERNAL']['path_dataset']
    generator = DatasetGeneratorUreter.from_parser(output_path=external_path, subdata_hypergui_mapping = None)
    # need ot set the subdata_hypergui_mapping. not sure where that happens
    
    generator.copy_mitk_data()
    generator.copied_mitk = True
    generator._paths = None #we need to reset the paths here if we want to apply out MITK filter
    #this setup may seem confusing, because it is  
    generator.run_safe(generator.dataset_settings)
    
    #this calls hypergui_labe_mapping, which calls yield hypergui
    paths_dict = generator.compile_image_dict() #dictionary where each value is a list of paths sharing the same image.
    list_of_paths_lists = list(paths_dict.values())
    p_map(generator.segmentations, list_of_paths_lists, num_cpus=2.0, task_name="Segmentation files") #the second argument should be a list of lists: a list of paths lists, each of which is the 
    generator.meta_table()
    #generator.median_spectra_table() #cant do because we dont have the proprietary function
    generator.preprocessed_files()   #also has a proprietary component
    #generator.view_organs()