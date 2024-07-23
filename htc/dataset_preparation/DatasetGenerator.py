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
from typing_extensions import Self

from htc.data_processing.run_l1_normalization import L1Normalization
from htc.data_processing.run_median_spectra import MedianSpectra
from htc.data_processing.run_parameter_images import ParameterImages
from htc.data_processing.run_raw16 import Raw16
#from htc.data_processing.run_rgb_sensor_aligned import RGBSensorAligned
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.tivita.metadata import generate_metadata_table
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.general import clear_directory, safe_copy
#from htc.utils.mitk.mitk_masks import nrrd_mask
from htc.utils.parallel import p_map
from htc.utils.paths import filter_min_labels


class DatasetGenerator:
    def __init__(
        self,
        output_path: Path,
        input_path: Path = None,
        paths: Union[list[DataPath], None] = None,
        regenerate: bool = False,
        include_csv: bool = False,
    ):
        """
        Basic class for our dataset reformatting (bringing the chaos structure provided by the clinicians into a suitable format). The basic idea is that there is a script for each dataset which contains all the steps of the processing pipeline. In the end, the data and the intermediates folder of the dataset should be created.

        This class provides some basic functionality shared between the different dataset scripts. You can use common attributes like `paths` (to get all images which should be processed) or common functions (e.g. meta table computation). For everything else, please create custom methods in your dataset script file or override existing ones.

        Args:
            output_path: Path where the reformatted dataset and intermediates should be stored (data and intermediates subfolder will be created at this location).
            input_path: Path to the original dataset from the clinicians (e.g. folders with the name `Cat_*`).
            paths: Explicit list of paths which should be considered in the methods. If None, all found images in the output path will be used (usually what you want).
            regenerate: If True, will delete existing intermediate files before generating new ones.
            include_csv: If True, will save tables additionally as CSV files.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = output_path.name
        self.include_csv = include_csv
        self._paths = paths if paths is not None else None

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
        Dictionary with dataset name as key and list of paths as value. Per default, this is a dictionary with only one entry (the main dataset) and paths of all images in the dataset. However, if the dataset contains subdatasets (e.g. 2021_02_05_Tivita_multiorgan_semantic and context_experiments), then this dictionary can be used to define the paths per subdataset (needed for the median table per subdataset). For the semantic dataset, this would look like:
        ```python
        {
            "2021_02_05_Tivita_multiorgan_semantic": [path1, path2, ...],
            "context_experiments": [path1, path2, ...],
        }
        ```
        In those cases, the generator class should overwrite the `dataset_paths` property.
        """
        paths = list(DataPath.iterate(self.data_dir))
        if (overlap_dir := self.data_dir / "overlap").exists():
            paths += list(DataPath.iterate(overlap_dir))

        return {self.output_path.name: paths}

    @property
    def paths(self) -> list[DataPath]:
        """
        List of all data paths (images) in the newly structured data directory (based on all paths defined by `dataset_paths`, i.e. including paths from subdatasets). This list can be used to create all the necessary intermediate files.
        """
        if self._paths is None:
            self._paths = list(itertools.chain(*self.dataset_paths.values()))

        return self._paths

    @paths.setter
    def paths(self, paths: list[DataPath]):
        self._paths = paths

    @property
    def tables_dir(self) -> Path:
        """
        Directory in the intermediates which contains the different tables.
        """
        target_dir = self.intermediates_dir / "tables"
        target_dir.mkdir(parents=True, exist_ok=True)

        return target_dir

    def meta_table(self) -> None:
        """
        segmentations should be generated before meta_table
        Creates and stores the meta table (e.g. `2021_02_05_Tivita_multiorgan_semantic@meta.feather`) for the dataset with all the available meta information per image (e.g. annotation names, camera meta). This table is necessary for the `DataPath.from_image_name()` functionality to work.
        """
        df = generate_metadata_table(track(self.paths, description="Collect metadata", refresh_per_second=1))
        df["path"] = [str(p().relative_to(self.data_dir)) for p in self.paths] # problem line!!!

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

    def median_spectra_table(
        self, recalibration_match: Union[Callable, None] = None, oversaturation_threshold: Union[int, None] = None
    ) -> None:
        """
        Creates and stores the median spectra table (e.g. `2021_02_05_Tivita_multiorgan_semantic@median_spectra@semantic#inter1.feather`) for the dataset with median spectra per label and image. A table per annotation name and subdataset will be created. You can use the `median_table()` function to easily read the table information later.

        Optionally, additional tables may be created based on different spectra preprocessing (e.g. recalibration). Each additional table gets their own table name (e.g. `2021_02_05_Tivita_multiorgan_semantic@recalibrated@median_spectra@semantic#inter1.feather`).

        Compared to the meta table, there are now multiple rows per image since one image usually contains more than one label.

        The median table contains the metadata as well but is larger (due to the spectra) and hence takes longer to read.

        Args:
            recalibration_match: If not None, will use the callable to recalibrate every path and use the corresponding matches for the table. This creates a new median table (using `recalibrated` as table_name).
            oversaturation_threshold: If not None, will apply the oversaturation threshold and exclude all spectra which are above the threshold. This creates a new median table (using `oversaturated-<oversaturation_threshold>` as table_name).
        """
        # Split median table by sub-datasets
        # A sub-dataset should not have an effect on the existing tables so we create a new one with a new dataset name (which is a subset name of the main dataset)
        for name, paths in self.dataset_paths.items():
            table_name = []
            if not any(filter_min_labels(p) for p in paths):
                continue

            # replace paths with recalibrated paths
            if recalibration_match is not None:
                table_name.append("recalibrated")
                paths_recalibrated = []
                for p in paths:
                    p_recalibrated = recalibration_match(p)
                    if p_recalibrated is not None:
                        paths_recalibrated.append(p_recalibrated)

                paths = paths_recalibrated

            if oversaturation_threshold is not None:
                table_name.append(f"oversaturated-{oversaturation_threshold}")

            if len(paths) == 0:
                continue
            else:
                table_name = ("_").join(table_name)
                if name != self.dsettings["dataset_name"]:
                    name = f"{self.dsettings['dataset_name']}#{name}"

                MedianSpectra(
                    paths,
                    dataset_name=name,
                    table_name=table_name,
                    output_dir=self.tables_dir,
                    oversaturation_threshold=oversaturation_threshold,
                ).run()

        if self.include_csv:
            for table_path in sorted(self.tables_dir.glob("*median_spectra*.feather")):
                df = pd.read_feather(table_path)
                df.to_csv(table_path.with_suffix(".csv"), index=False)

    def colorchecker_median_spectra_table(
        self, recalibration_match: Union[Callable, None] = None, annotation_name: str = "squares#automask"
    ) -> None:
        """
        Computes a global colorchecker median spectra table based on all colorchecker images.

        Args:
            recalibration_match: If not None, will use the callable to recalibrate every path and use the corresponding matches for the table. This creates a new median table (using `recalibrated` as table_name).
            annotation_name: The annotation name which must be used for the colorchecker images. Please note that this is not a selection but only ensures that all colorchecker images use the same annotation name.
        """
        for name, paths in self.dataset_paths.items():
            paths_cc = [p for p in paths if p.colorchecker_annotation_path() is not None]
            if not any(paths_cc):
                continue

            assert all(
                annotation_name in p.colorchecker_annotation_path().stem for p in paths_cc
            ), f"The colorchecker images do not have the annotation name {annotation_name}"

            # Replace paths with recalibrated paths
            table_name = ["colorchecker"]
            if recalibration_match is not None:
                table_name.append("recalibrated")
                paths_recalibrated = []
                for p in paths_cc:
                    p_recalibrated = recalibration_match(p)
                    if p_recalibrated is not None:
                        paths_recalibrated.append(p_recalibrated)

                paths_cc = paths_recalibrated

            if len(paths) == 0:
                continue
            else:
                table_name = ("_").join(table_name)
                if name != self.dsettings["dataset_name"]:
                    name = f"{self.dsettings['dataset_name']}#{name}"

                tables = p_map(
                    self._colorchecker_table_helper,
                    paths_cc,
                    task_name="Colorchecker median table",
                    num_cpus=2.0,
                )
                df_cc = pd.concat(tables, ignore_index=True)

                # Similar to the median spectra table, we directly add the metadata to the table
                df_meta = pd.read_feather(self.tables_dir / f"{self.dsettings['dataset_name']}@meta.feather")
                df_cc = df_cc.merge(df_meta, how="left")
                df_cc["annotation_name"] = annotation_name

                target_file = self.tables_dir / f"{name}@{table_name}@median_spectra@{annotation_name}.feather"
                df_cc.to_feather(target_file)
                settings.log.info(
                    f"Wrote {'recalibrated ' if recalibration_match is not None else ''}colorchecker median table to"
                    f" {target_file}"
                )

    def _colorchecker_table_helper(self, path: DataPath) -> pd.DataFrame:
        cc_mask = path.read_colorchecker_mask()
        assert cc_mask is not None, f"Colorchecker mask for {path.image_name()} does not exist yet."

        # Add metadata which we need for the global table
        cc_mask = cc_mask["median_table"]
        cc_mask["image_name"] = path.image_name()

        return cc_mask

    def preprocessed_files(self, include_raw16: bool = True, recalibration_match: Union[Callable, None] = None) -> None:
        """
        Computes L1 normalized cubes, raw16 cubes and parameter images for every image in the dataset.

        Args:
            include_raw16: If True, will also compute the raw16 files.
            recalibration_match: If not None, will use the recalibrated paths for the preprocessing (stored in their own folder, e.g., L1_recalibrated).
        """
        if recalibration_match is not None:
            folder_suffix = "_recalibrated"
            paths = []
            for p in self.paths:
                p_recalibrated = recalibration_match(p)
                if p_recalibrated is not None:
                    paths.append(p_recalibrated)
        else:
            folder_suffix = ""
            paths = self.paths

        if len(paths) > 0:
            L1Normalization(
                paths,
                file_type="blosc",
                output_dir=self.intermediates_dir / "preprocessing" ,
                regenerate=self.regenerate,
                #folder_name=f"L1{folder_suffix}",
            ).run()
        """
            ParameterImages(
                paths,
                file_type="blosc",
                output_dir=self.intermediates_dir / "preprocessing",
                regenerate=self.regenerate,
                #folder_name=f"parameter_images{folder_suffix}",
            ).run()

            if include_raw16:
                Raw16(
                    paths,
                    file_type="blosc",
                    output_dir=self.intermediates_dir / "preprocessing",
                    regenerate=self.regenerate,
                    folder_name=f"raw16{folder_suffix}",
                ).run()
        """

    def aligned_rgb_sensor(self) -> None:
        RGBSensorAligned(
            self.paths,
            file_type="blosc",
            output_dir=self.intermediates_dir / "preprocessing",
            regenerate=self.regenerate,
        ).run(num_cpus=4)

    def _add_temperatures(self, df_meta: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sensor and light temperature information to the metadata table.

        This uses the temperature information stored in the meta log files. The average between the temperature before and after image acquisition is computed. For each camera, images with 0 temperature values (happens if the camera cannot store the metadata successfully) are interpolated based on all other images from this camera. Only interpolation and not extrapolation is performed. Please note that not every camera stores this information, so nan values are expected for older images.

        Args:
            df_meta: The table with the metadata.

        Returns: A new metadata table with `sensor_temperature` and `light_temperature` columns.
        """
        df_meta.set_index("image_name", inplace=True)

        for target_column, name_before, name_after in [
            ("sensor_temperature", "Temperaturen_HSI-Sensor Temp. vor Scan", "Temperaturen_HSI-Sensor Temp. nach Scan"),
            ("light_temperature", "Temperaturen_LED Temp. vor Scan", "Temperaturen_LED Temp. nach Scan"),
        ]:
            df_meta[target_column] = pd.NA

            # We only operate on images which store the temperature (this is not the case for older cameras)
            df_valid = df_meta[(~pd.isna(df_meta[name_before])) & (~pd.isna(df_meta[name_after]))]

            # Temperature is camera-specific, so we do the interpolations also per camera
            for cam in df_valid["Camera_CamID"].unique():
                df_cam = df_valid[df_valid["Camera_CamID"] == cam]

                # Sometimes, one of the two temperature values is missing. We fill the missing values based on the global difference between the two temperatures (per camera)
                df_cam_both = df_cam[(df_cam[name_before] > 0) & (df_cam[name_after] > 0)]
                global_cam_diff = np.median(df_cam_both[name_after] - df_cam_both[name_before])

                # Collect temperature values for the interpolation
                times = []
                temps = []
                for _, row in df_cam.iterrows():
                    if row[name_before] == 0 and row[name_after] == 0:
                        continue

                    time_path = pd.to_datetime(row["timestamp"], format=settings.tivita_timestamp_format).timestamp()
                    temp_before = row[name_before] if row[name_before] > 0 else row[name_after] - global_cam_diff
                    temp_after = row[name_after] if row[name_after] > 0 else row[name_before] + global_cam_diff

                    times.append(time_path)
                    temps.append((temp_before + temp_after) / 2)

                times = np.asarray(times)
                temps = np.asarray(temps)

                # The arrays must be sorted for the interpolation
                indices = np.argsort(times)
                times = times[indices]
                temps = np.take(temps, indices)

                assert np.all(np.diff(times) > 0), (
                    "The time axis must be strictly increasing and must not contain any duplicate timestamps (camera:"
                    f" {cam})"
                )

                # Obtain interpolated temperature value for each image
                for image_name, row in df_cam.iterrows():
                    time_path = pd.to_datetime(row["timestamp"], format=settings.tivita_timestamp_format).timestamp()
                    temp = np.interp(time_path, times, temps, left=np.nan, right=np.nan)
                    df_meta.loc[image_name, target_column] = temp

        df_meta = df_meta.reset_index()
        df_meta = df_meta.convert_dtypes()
        return df_meta

    def _copy_hypergui_data(
        self,
        paths: list[DataPath],
        target_dir: Path,
        hypergui_mapping: dict[str, dict[str, str]],
    ) -> None:
        """
        Copy data and annotations from a folder with hypergui annotations. This method copies the image data, creates PNG mask files for the annotations and fills the meta label JSON file.

        Note: Image data is copied from the first path in the list if the `*SpecCube.dat` is missing in the target directory.

        The intended usage of this function is to call it for every target image in your dataset:
        ```python
        for folder in unstructured_dataset:
            self._copy_hypergui_paths(...)

        # Warning report summary from all copied paths
        self._check_hypergui_paths()
        ```

        Args:
            paths: List of image paths which all show the same image but annotate different parts in the image (since an image may be stored more than once in the original dataset).
            target_dir: Path to the directory in the structured dataset for this image.
            hypergui_mapping: Information for every _hypergui_N folder. For each folder, a dictionary with the keys `annotation_type` (e.g. polygon or circle) and `annotator` (e.g. annotator1) must be provided.
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

        # Subfolder with everything which we know about the data
        target_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir = target_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        self.missing_binaries = []
        self.missing_coordinates = []

        timestamp = target_dir.name
        meta_labels_path = annotations_dir / f"{timestamp}_meta.json"
        if meta_labels_path.exists():
            with meta_labels_path.open() as f:
                meta_labels = json.load(f)
        else:
            meta_labels = {"image_labels": []}
        meta_labels_copy = copy.deepcopy(meta_labels)

        ignored_prefixes = (".", "_", "Thumbs.db")

        def _extend_meta_labels(label: str) -> None:
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
            # Copy missing image data
            # overlap is an exception for the masks and semantic dataset where image files are primary stored in the semantic dataset. This should not be used in the future anymore (use the annotation_name feature instead)
            if "overlap" not in str(target_dir) and len(sorted(target_dir.glob("*SpecCube.dat"))) == 0:
                for f in sorted(p().iterdir()):
                    if f.is_file() and not f.name.startswith(ignored_prefixes):
                        if f.suffix != ".xml" and not f.stem.startswith(timestamp):
                            settings.log.error(
                                f"The timestamp of the folder {timestamp} and the file name {f.name} do not match"
                                " (every file should start with the timestamp of the folder)"
                            )
                        safe_copy(f, target_dir / f.name)

            # The path is only associated with one label
            if hasattr(p, "organ"):
                label = p.organ
                _extend_meta_labels(label)

            for hypergui_dir in sorted(p().iterdir()):
                if hypergui_dir.is_dir() and hypergui_dir.name.startswith("_hypergui"):
                    if hypergui_dir.name not in hypergui_mapping:
                        settings.log.warning(
                            f"Found an unknown hypergui folder: {hypergui_dir}. The folder will be ignored"
                        )
                        continue

                    annotation_info = hypergui_mapping[hypergui_dir.name]
                    if "label_name" in annotation_info:
                        # The path is associated with multiple labels (each label with their own _hypergui_* folder)
                        label = annotation_info["label_name"]
                        _extend_meta_labels(label)

                    binary_path = hypergui_dir / "mask.csv"
                    if binary_path.exists():
                        mask = pd.read_csv(binary_path, header=None).to_numpy()
                        mask = mask.astype(bool)
                        mask = Image.fromarray(mask).convert("1")

                        mask.save(
                            annotations_dir
                            / f"{p.timestamp}#{annotation_info['annotation_type']}#{annotation_info['annotator']}#{label}#binary.png",
                            optimize=True,
                        )
                    else:
                        self.missing_binaries.append(hypergui_dir)

                    coordinates_path = hypergui_dir / "MASK_COORDINATES.csv"
                    if coordinates_path.exists():
                        safe_copy(
                            coordinates_path,
                            annotations_dir
                            / f"{p.timestamp}#{annotation_info['annotation_type']}#{annotation_info['annotator']}#{label}#coordinates.csv",
                        )
                    else:
                        self.missing_coordinates.append(hypergui_dir)

        if meta_labels != meta_labels_copy:
            with meta_labels_path.open("w") as f:
                json.dump(meta_labels, f, cls=AdvancedJSONEncoder, indent=4)

    def _check_hypergui_data(self) -> None:
        assert (
            self.missing_binaries is not None and self.missing_coordinates is not None
        ), "Please call _copy_hypergui_paths() before running this check"

        if len(self.missing_binaries) > 0 or len(self.missing_coordinates) > 0:
            settings.log.warning(f"There are {len(self.missing_binaries)} _hypergui folders without a mask.csv file")
            settings.log.warning(
                f"There are {len(self.missing_coordinates)} _hypergui folders without a MASK_COORDINATES.csv file"
            )

            bin_coord = set(self.missing_binaries) - set(self.missing_coordinates)
            msg = f"{len(bin_coord)} folders with binaries but no coordinates:\n"
            for f in bin_coord:
                msg += f"\t{f}\n"

            coord_bin = set(self.missing_coordinates) - set(self.missing_binaries)
            msg += f"{len(coord_bin)} folders with coordinates but no binaries:\n"
            for f in coord_bin:
                msg += f"\t{f}\n"

            intersection = set(self.missing_binaries).intersection(set(self.missing_coordinates))
            msg += f"{len(intersection)} folders self.missing both:\n"
            for f in intersection:
                msg += f"\t{f}\n"
            settings.log.warning(msg)

    def _hypergui_label_mapping(self) -> tuple[dict[str, int], int]:
        """
        Crawls the dataset and searches for all annotated labels, i.e. labels with annotations (png files). Labels which only occur as image labels are not considered.

        Returns: A label mapping and the index of the last valid label (i.e. the last label which is neither overlap nor unlabeled).
        """
        # Get all labels in the dataset
        all_labels = set()
        for path in track(self.paths, description="Collect labels", refresh_per_second=1):
            image_labels = path.meta("image_labels")
            assert image_labels is not None, f"There must always be at least image labels ({path})"

            # Only labels that are annotated in the image are relevant for the label mapping
            for png_file in sorted((path() / "annotations").glob("*.png")):
                timestamp, annotation_type, annotator, label_name, file_type = png_file.stem.split("#")
                assert timestamp == path.timestamp
                assert annotation_type == "polygon"
                assert label_name in image_labels, (
                    f"The label {label_name} is annotated for the image {path} but the label is not part of the"
                    f" image label list: {image_labels}"
                )
                assert file_type == "binary"

                all_labels.add(label_name)

            for nrrd_file in sorted((path() / "annotations").glob("*.nrrd")):
                timestamp, annotation_type, annotator = nrrd_file.stem.split("#")
                assert timestamp == path.timestamp

                mitk_data = nrrd_mask(nrrd_file)
                labels = mitk_data["label_mapping"].label_names()
                assert set(labels).issubset(
                    image_labels
                ), f"Labels {labels} are not part of the image labels: {image_labels}"
                all_labels.update(labels)

        # Generate a mapping based on the sorted label list
        all_labels = sorted(all_labels)
        label_mapping = dict(zip(all_labels, range(len(all_labels))))
        assert len(label_mapping) < 254, "Too many labels"

        last_valid_label_index = len(label_mapping) - 1
        label_mapping["overlap"] = 254
        label_mapping["unlabeled"] = 255

        return label_mapping, last_valid_label_index

    @classmethod
    def from_parser(
        cls,
        output_path=None,
        include_csv: bool = False,
        additional_arguments: dict[str, dict[str, Any]] = None,
        **kwargs,
    ) -> Self:
        """
        Factory method to create an instance of your dataset generation class with common command line options.

        Args:
            output_path: Path where the structured dataset should be stored. If None, path must be given on the command line.
            include_csv: If True, store tables additionally as CSV.
            additional_arguments: Additional command line options. Keys of the dict are the names of the argument (first parameter to the `parser.add_argument()` method) and values are dictionaries with the remaining keyword arguments.

        Returns: Instance of your dataset generation class initialized command line arguments.
        """
        parser = argparse.ArgumentParser(
            description=(
                "Reformat the dataset (from the format provided by clinicians) and creates all the intermediate files"
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        if output_path is None:
            parser.add_argument(
                "--output-path",
                type=Path,
                required=True,
                help=(
                    "Path where the reformatted dataset and intermediates should be stored (data and intermediates"
                    " subfolder will be created at this location)."
                ),
            )
        else:
            parser.add_argument(
                "--output-path",
                type=Path,
                default=output_path,
                help=(
                    "Path where the reformatted dataset and intermediates should be stored (data and intermediates"
                    " subfolder will be created at this location)."
                ),
            )

        parser.add_argument(
            "--regenerate",
            action="store_true",
            default=False,
            help="If set, will delete an existing intermediates folder to regenerate all files.",
        )
        parser.add_argument(
            "--include-csv",
            action="store_true",
            default=include_csv,
            help="If set, will also store tables as CSV.",
        )

        parser.add_argument(
            "--input-path",
            type=Path,
            nargs='+',
            required=False,
            help="Path to the original dataset from the clinicians (e.g. folders with the name `Cat_*`).",
        )

        if additional_arguments is not None:
            for name, arguments in additional_arguments.items():
                parser.add_argument(name, **arguments)

        args = vars(parser.parse_args()) | kwargs
        return cls(**args)

    @staticmethod
    def run_safe(func: callable, *args, **kwargs) -> None:
        """
        Run a function in a separate process to avoid caching issues.

        Methods of this class may fill the DataPath caches with old data, e.g. when executing `dataset_settings()` the `dataset_settings` cache will be filled with the old data (due to the usage of `path.meta()`). During execution, the `dataset_settings()` method may adds new data or modifies existing data of the `dataset_settings`. However, due to the caching, the new data will not be used in future methods. To avoid this issue, it is recommended to run those methods which alter information of the caches in a separate process.

        Args:
            func: Function which should be executed in a separate process.
            args: Arguments for the function.
            kwargs: Keyword arguments for the function.
        """
        p = Process(target=func, args=args, kwargs=kwargs)
        p.start()
        p.join()
