import argparse
import copy
import itertools
import json
import re
from functools import cached_property
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Union
import numpy as np
import pandas as pd
from PIL import Image
from rich.progress import track
from typing_extensions import Self, Generator, list, dict, tuple
from htc.tivita.DataPathAtlas2 import DataPathAtlas
from htc.data_processing.run_l1_normalization import L1Normalization
from htc.data_processing.run_median_spectra import MedianSpectra
from htc.data_processing.run_parameter_images import ParameterImages
from htc.data_processing.run_raw16 import Raw16
from htc.data_processing.run_rgb_sensor_aligned import RGBSensorAligned
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.dataset_preparation.DatasetGenerator import DatasetGenerator
from htc.tivita.DatasetSettings import DatasetSettings
from htc.tivita.metadata import generate_metadata_table
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.general import clear_directory, safe_copy
from htc.utils.mitk.mitk_masks import nrrd_mask
from htc.utils.parallel import p_map
from htc.utils.paths import filter_min_labels
from htc.utils.LabelMapping import LabelMapping
from htc.utils.blosc_compression import compress_file

class DatasetGeneratorAlex(DatasetGenerator):
    def __init__(self, output_path: Path, input_paths: list[Path] = None, paths: list[DataPath] | None = None, regenerate: bool = False, include_csv: bool = False):
    
        """
        Class to process data from Alex in its original form. 
        Like other DatasetGenerator classes, this class uses hte base parent class and is designed to process
        a dataset given from the clinicians (alex). However, Unlike the other DatasetGenerators,
        This class DOES NOT COPY the dataset into the new structure. Instead, it leaves it in place and creates a new
        location only for the new, generated/preprocessed data/information. that eternal directory can be then
        defined as an environment variable, which preserves the functionality of the original htc approach

        not currently capable of handling MITK annotations

        Args:
            output_path: Path where the intermediates and dataset_settings should be stored (data and intermediates subfolder will be created at this location).
            input_paths: list of Paths to the original dataset from the clinicians (e.g. folders with the name `Cat_*`).
            paths: Explicit list of paths which should be considered in the methods. If None, all found images in the output path will be used (usually what you want).
            regenerate: If True, will delete existing intermediate files before generating new ones.
            include_csv: If True, will save tables additionally as CSV files.
        """
        self.input_paths = input_paths
        self.output_path = output_path
        self.dataset_name = output_path.name
        self.include_csv = include_csv
        self._paths = paths if paths is not None else None
        self.hypergui_subdata_mapping = None #need to add this in
        self.data_dir = self.output_path / "data"
        if not self.data_dir.exists():
            self.data_dir.mkdir(exist_ok=True, parents=True)
        else:
            assert self.data_dir.is_dir(), (
                "There is a file named data in the output_path. Please remove the file, as the script needs to create a"
                " data directory."
            )

        if self.input_path is not None:
            assert self.input_path.is_dir(), "The input_path does not point towards a directory."

        self.intermediates_dir = self.output_path / "intermediates"
        if not self.intermediates_dir.exists():
            self.intermediates_dir.mkdir(exist_ok=True, parents=True)

            # Make the newly created directory available so that scripts can use it
            settings.intermediates_dir_all.add_alternative(self.intermediates_dir)

        if regenerate:
            clear_directory(self.intermediates_dir)

        self.regenerate = regenerate
        self.dsettings = DatasetSettings(self.data_dir)

        self.missing_binaries = None
        self.missing_coordinates = None
        
    
    
    @cached_property
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
        dict = {}
        data_subdirectories = [path / 'data' for path in self.input_paths]
        for data_dir in data_subdirectories:
            paths = list(DataPathAtlas.iterate(data_dir))
            name0 = paths[0].subdataset_name
            assert all(path.subdataset_name == name0 for path in paths), "Not all items are from the same subdataset"
            dict[name0] = paths #set the ist of paths to
            
        return dict
        
    
    def compile_image_dict(self):
        """
        method to take the dataset_paths dictionary, and produce a dictionary with the unique image names as keys,
        and the Paths leading to that image name as values. This is intended as a tool to deal with duplicates in the larger dataset, particular ones
        found in different "data" folders within the larger dataset
        
        Returns:
            Dict: dictionary of key-value pairs - str: list[DataPath] - the string is the unique image_name of the image. the list is a list
            of datapath objects to each of the different locations where the duplicate exists. 
        """
        images_dict = {}
        for path in self.paths:
            name = path.image_name()
            images_dict.setdefault(name, []).append(name) #check to see if a list has already been created. if not, initialize it
        return images_dict
    
    def subdatasets(self):
        """another heping method to organize the dataset. takes in a list of paths, and creates a dictionary
        with key value pairs:
            "name_of_subdataset" : Paths in that subdataset
        here, a subdataset is defined as a directory immediately containing a "data" folder in the original
        unstructured dataset.
        should e used INSTEAD of compile_image_dict, depending on context, because they both divide along the same
        scope, just using different criteria. 
        """
        subdatasets = {}
        for path in self.paths:
            name = path.subdataset_name #here, name is the name of the subdataset folder
            subdatasets.setdefault(name, []).append(path)#appends to list for that subdataset (createsif it does not yet exist)
        return subdatasets
    
    
    def dataset_settings(self) -> None:
        label_mapping, last_valid_label_index = self._hypergui_label_mapping()

        dataset_settings = {
            "dataset_name": "2021_02_05_Tivita_multiorgan_masks",
            "data_path_class": "htc.tivita.DataPathMultiorgan>DataPathMultiorgan",
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


    def segmentations(self, paths):
        """
        method to iterate through all the hyperGUI folders in all the DataPaths available (across all subdatasets),
        and compress them into a blosc dictionary in the standard format of the intermediates directory.

        Args:
            paths: list of paths, all of which must correspond to the same image. 
        """
        label_mapping = LabelMapping.from_data_dir(self.data_dir)
        target_dir = self.intermediates_dir / "segmentations"
        target_dir.mkdir(parents=True, exist_ok=True)

       

        annotations = {}
        for dictionary in self._yield_hypergui_data(paths, self.subdata_hypergui_mapping):
            timestamp = dictionary['timestamp']
            annotation_type = dictionary['annotation_type']
            annotator = dictionary['annotator']
            label_name = dictionary['label']
            assert label_name in label_mapping, f"No label_index defined for the label {label_name}"
            # assert timestamp == path.timestamp

            # Read mask
            mask = np.array(dictionary['mask'])
            assert mask.dtype == bool, f"The mask must have a boolean dtype: {f}"
            assert 1 <= len(np.unique(mask)) <= 2, f"Invalid number of values in the mask {f}"
            assert mask.shape == self.dsettings["spatial_shape"], f"The mask has the wrong shape: {mask.shape}"

            annotation_name = f"{annotation_type}#{annotator}"

            if annotation_name not in annotations:
                annotations[annotation_name] = np.full(
                    self.dsettings["spatial_shape"], label_mapping.name_to_index("unlabeled"), dtype=np.uint8
                )

            if not np.all(annotations[annotation_name][mask] == label_mapping.name_to_index("unlabeled")):
                labels = [label_mapping.index_to_name(l) for l in np.unique(annotations[annotation_name][mask])]
                labels = [l for l in labels if l != "unlabeled"]

                settings.log.error(
                    "Overlapping pixel detected for the image"
                    f" {path.image_name()} (annotation_name={annotation_name}, new label={label_name}, existing"
                    f" labels={labels}). The annotation for the new label {label_name} will be ignored!"
                )
            else:
                annotations[annotation_name][mask] = label_mapping.name_to_index(label_name)
            
            """
            elif f.suffix == ".nrrd":
                timestamp, annotation_type, annotator = f.stem.split("#")
                assert timestamp == path.timestamp

                mitk_data = nrrd_mask(f)

                assert set(mitk_data["label_mapping"].label_names()).issubset(label_mapping.label_names()), (
                    f"The file {f} contains the labels"
                    f" {set(mitk_data['label_mapping'].label_names()) - set(label_mapping.label_names())} that are not"
                    " defined in the label mapping of the dataset"
                )
                mask = label_mapping.map_tensor(mitk_data["mask"], mitk_data["label_mapping"])

                if mask.ndim == 3:  # for multi layer nrrd files
                    layer_to_type = path.dataset_settings[f"layer_to_type{mask.shape[0]}"]

                    for layer in range(mask.shape[0]):
                        annotation_name = f"{layer_to_type[str(layer)]}#{annotator}"
                        annotations[annotation_name] = mask[layer, :, :]
                else:
                    annotation_name = f"{annotation_type}#{annotator}"
                    annotations[annotation_name] = mask
                """
        if len(annotations) > 0:
            compress_file(target_dir / f"{paths[0].image_name()}.blosc", annotations)


            


    def meta_table(self) -> None:
        """
        segmentations should be generated before meta_table
        Creates and stores the meta table (e.g. `2021_02_05_Tivita_multiorgan_semantic@meta.feather`) for the dataset with all the available meta information per image (e.g. annotation names, camera meta). This table is necessary for the `DataPath.from_image_name()` functionality to work.
        """
        df = generate_metadata_table(track(self.paths, description="Collect metadata", refresh_per_second=1))
        df["path"] = [str(p.grandparent/'data') for p in self.paths] # problem line!!! maybe fixed with grandparent

        if len(self.dataset_paths) > 1:
            # Also store the path to the corresponding dataset_settings for each path (especially important for subdatasets)
            df["dataset_settings_path"] = [
                str(p.dataset_settings.settings_path.relative_to(self.data_dir)) for p in self.paths #not a problkem line? just sets dataset_settings path, which is in the right place
            ]

        names = []
        for p in track(self.paths, description="Collect annotation names", refresh_per_second=1):
            seg = p.read_segmentation("all")
            if seg is not None:
                names.append(list(seg.keys()))
            else:
                names.append(None)
        df["annotation_name"] = names

        df = df.reset_index(drop=True)
        df.to_feather(self.tables_dir / f"{self.dsettings['dataset_name']}@meta.feather")

        if self.include_csv:
            df.to_csv(self.tables_dir / f"{self.dsettings['dataset_name']}@meta.csv", index=False)

    def view_organs(self) -> None:
        mapping = LabelMapping.from_data_dir(self.data_dir)
        missing_colors = []
        for label in mapping.label_names():
            if label not in settings.label_colors:
                missing_colors.append(label)

        assert (
            len(missing_colors) == 0
        ), f"The following labels do not have a color defined in settings.label_colors: {missing_colors}"

        p_map(
            partial(self._save_html, navigation_paths=self.paths),
            self.paths,
            task_name="View organs",
        )



    def _yield_hypergui_data(
        self,
        paths: list[DataPath],
        hypergui_mapping: dict[str, dict[str, str]], #want to change this
        )   -> Generator[dict, None, None]:
        """
        Yield data and annotations from a folder with hypergui annotations. This method processes the image data, generates PNG mask files for the annotations, and collects the meta label JSON information.

        Args:
            paths: List of image paths which all show the same image but annotate different parts in the image (since an image may be stored more than once in the original dataset).
            hypergui_mapping: Information for every _hypergui_N folder. For each folder, a dictionary with the keys `annotation_type` (e.g. polygon or circle) and `annotator` (e.g. annotator1) must be provided.

        Yields:
            dict: A dictionary containing the processed data for each path, including masks, meta labels, and any relevant annotations.
        """
        assert all(
            p.timestamp == paths[0].timestamp for p in paths
        ), "The timestamp for all image paths must be the same (identical image data)"

        # Angle mapping for the standardized dataset (only used if available)
        angle_mapping = {
            1: 0,
            2: -25,
            3: 25,
        }

        self.missing_binaries = []
        self.missing_coordinates = []

        ignored_prefixes = (".", "_", "Thumbs.db")

        def _extend_meta_labels(label: str, meta_labels: dict, p: DataPath) -> None:
            if label not in meta_labels["image_labels"]:
                meta_labels["image_labels"].append(label)
                meta_labels["image_labels"] = sorted(set(meta_labels["image_labels"]))

            for label_file in sorted(p().glob("_labelling*")):
                if "paper_tags" not in meta_labels:
                    meta_labels["paper_tags"] = []

                if label_file.stem not in meta_labels["paper_tags"]:
                    meta_labels["paper_tags"].append(label_file.stem)

                # Convert standardized information to a structured format (if available)
                match = re.search(r"_labelling_standardized_situs_(\d+)", label_file.stem)
                if match is not None:
                    situs = int(match.group(1))

                    if "label_meta" not in meta_labels:
                        meta_labels["label_meta"] = {}
                    label_meta = meta_labels["label_meta"]
                    if label not in label_meta:
                        label_meta[label] = {}
                    current_label_meta = label_meta[label]

                    if "situs" in current_label_meta:
                        assert (
                            current_label_meta["situs"] == situs
                        ), f"Found different situs for the label: {label} (path {p})"
                    else:
                        current_label_meta["situs"] = situs

                    match = re.search(
                        r"_labelling_standardized_situs_(\d+)_angle_(\d+)_repetition_(\d+)", label_file.stem
                    )
                    if match is not None:
                        angle = angle_mapping[int(match.group(2))]
                        repetition = int(match.group(3))
                        if "angle" in current_label_meta:
                            assert current_label_meta["angle"] == angle, f"Found different angle for the label: {label}"
                        else:
                            current_label_meta["angle"] = angle
                        if "repetition" in current_label_meta:
                            assert (
                                current_label_meta["repetition"] == repetition
                            ), f"Found different repetition for the label: {label}"
                        else:
                            current_label_meta["repetition"] = repetition
        
        for p in paths:
            meta_labels = {"image_labels": []}

            # The path is only associated with one label
            if hasattr(p, "organ"):
                label = p.organ
                _extend_meta_labels(label, meta_labels, p)

            for hypergui_dir in sorted(p().iterdir()): #goes through and assigns labels according to Hypergui mapping
                #might have to also change if there are multiple subdatasets under the same label.
                if hypergui_dir.is_dir() and hypergui_dir.name.startswith("_hypergui"):
                    subdata_hypergui_name = f"{p.subdataset_name}#{hypergui_dir.name}" #subdata_hypergui_name is a string in the format: "subdatasetname#_hyperGUI_N"
                    if subdata_hypergui_name not in hypergui_mapping:
                    # settings.log.warning(
                    #     f"Found an unknown hypergui folder: {hypergui_dir}. The folder will be ignored"
                    #)
                    #removed warning, since we expect to be skipping over hyperGUI folders
                        continue

                    annotation_info = hypergui_mapping[subdata_hypergui_name]
                    if "label_name" in annotation_info:
                        # The path is associated with multiple labels (each label with their own _hypergui_* folder)
                        label = annotation_info["label_name"]
                        _extend_meta_labels(label, meta_labels, p) #this updates the Datapath objects metadata to include the label

                    binary_path = hypergui_dir / "mask.csv"
                    if binary_path.exists():
                        mask = pd.read_csv(binary_path, header=None).to_numpy()
                        mask = mask.astype(bool)
                        mask = Image.fromarray(mask).convert("1")
                    else:
                        self.missing_binaries.append(hypergui_dir)
                        mask = None

                    coordinates_path = hypergui_dir / "MASK_COORDINATES.csv"
                    if not coordinates_path.exists():
                        self.missing_coordinates.append(hypergui_dir)
                        coordinates_path = None
                    yield {
                        "path": p,
                        "meta_labels": meta_labels,
                        "timestamp": p.timestamp,
                        "annotation_type": annotation_info["annotation_type"],
                        "annotator": annotation_info["annotator"],
                        "label": label,
                        "mask": mask,
                        "coordinates_path": coordinates_path,
                    }



    def _hypergui_label_mapping(self) -> tuple[dict[str, int], int]:
        """
        Crawls the dataset and searches for all annotated labels, i.e. labels with annotations (png files). Labels which only occur as image labels are not considered.

        Returns: A label mapping and the index of the last valid label (i.e. the last label which is neither overlap nor unlabeled).
        """
        # Get all labels in the dataset
        all_labels = set()
        
        for image_name, paths in enumerate(self.compile_image_dict()):
            for dictionary in self._yield_hypergui_data(paths, self.subdata_hypergui_mapping):
                label = dictionary['label'] 
                all_labels.add(label)
        
        
        
        # Generate a mapping based on the sorted label list
        all_labels = sorted(all_labels)
        label_mapping = dict(zip(all_labels, range(len(all_labels))))
        assert len(label_mapping) < 254, "Too many labels"

        last_valid_label_index = len(label_mapping) - 1
        label_mapping["overlap"] = 254
        label_mapping["unlabeled"] = 255

        return label_mapping, last_valid_label_index
        
        
    @classmethod
    def from_parser(cls, **kwargs) -> Self:
        if "additional_arguments" not in kwargs:
            kwargs["additional_arguments"] = {}

        kwargs["additional_arguments"] |= {
            "--subdata-hypergui-mapping": {
                "type": Path,
                "required": True,
                "default": None,
                "help": "Path to the subdata-hypergui-mapping JSON, which should consist of a dictionary of 'subdataname#_hypergui_N' : 'label' key-value pairs"
            },
        }

        return super().from_parser(**kwargs)  
        
if __name__ == "__main__":
    # The path argument for DatasetGenerator can be set to 2021_02_05_Tivita_multiorgan_masks dataset paths. Example:
    # screen -S dataset_masks -d -m script -q -c "htc dataset_masks --input-path /mnt/E130-Projekte/Biophotonics/Data/2020_07_23_hyperspectral_MIC_organ_database/data/Catalogization_tissue_atlas" dataset_masks.log
    external_path = settings.external_dir["dataset_path"]
    generator = DatasetGeneratorAlex.from_parser(output_path=external_path)
    # need ot set the subdata_hypergui_mapping. not sure where that happens
    
    
    generator.run_safe(generator.dataset_settings)
    #this calls hypergui_labe_mapping, which calls yield hypergui

    p_map(generator.segmentations, generator.paths, num_cpus=2.0, task_name="Segmentation files")
    generator.meta_table()
    generator.median_spectra_table()
    generator.preprocessed_files()
    generator.view_organs()