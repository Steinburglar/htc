import copy
import json
import re
from functools import cached_property
from pathlib import Path
from typing import Union

import numpy as np
from rich.progress import track
from typing_extensions import Self

from htc.cameras.calibration.CalibrationSwap import recalibrated_path
from htc.dataset_preparation.run_dataset_masks import DatasetGeneratorMasks
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.general import safe_copy
from htc.utils.helper_functions import sort_labels
from htc.utils.mitk.mitk_masks import nrrd_mask
from htc.utils.parallel import p_map


class DatasetGeneratorRat(DatasetGeneratorMasks):
    def __init__(
        self,
        *args,
        input_path_halogen: Path = None,
        input_path_led: Path = None,
        input_path_straylight: Path = None,
        input_path_perfusion: Path = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.input_path_halogen = input_path_halogen
        if self.input_path_halogen is not None:
            assert self.input_path_halogen.is_dir(), "The input_path_halogen does not point towards a directory."
        self.input_path_led = input_path_led
        if self.input_path_led is not None:
            assert self.input_path_led.is_dir(), "The input_path_led does not point towards a directory."
        self.input_path_straylight = input_path_straylight
        if self.input_path_straylight is not None:
            assert self.input_path_straylight.is_dir(), "The input_path_straylight does not point towards a directory."
        self.input_path_perfusion = input_path_perfusion
        if self.input_path_perfusion is not None:
            assert self.input_path_perfusion.is_dir(), "The input_path_perfusion does not point towards a directory."

        # So far only two polygon annotations available
        n_hypergui_folder = 2
        self.hypergui_mapping = {}
        for r in range(1, n_hypergui_folder + 1):
            self.hypergui_mapping[f"_hypergui_{r}"] = {
                "annotation_type": "polygon",
                "annotator": f"annotator{r}",
            }

        self.label_renaming = {
            "subcut_fat": "fat_subcutaneous",
            "kidney_with_Ger_fascia": "kidney_with_Gerotas_fascia",
            "visc_fat": "fat_visceral",
        }

        self.overview_kwargs = {"nav_width": "31em"}

    @cached_property
    def dataset_paths(self) -> dict[str, list[DataPath]]:
        return {
            self.output_path.name: list(DataPath.iterate(self.data_dir)),
            "straylight_experiments": list(DataPath.iterate(self.data_dir / "straylight_experiments")),
            "perfusion_experiments": list(DataPath.iterate(self.data_dir / "perfusion_experiments")),
        }

    @cached_property
    def paths_halogen(self) -> list[DataPath]:
        assert (
            self.input_path_halogen is not None
        ), "--input-path-halogen cannot be None, when calling the copy_*_data() functions"
        paths = list(DataPath.iterate(self.input_path_halogen))

        for p in paths:
            p.subject_name = p.subject_name.replace("P", "R")
            p.organ = self.label_renaming.get(p.organ, p.organ)

        return paths

    @cached_property
    def paths_led(self) -> list[DataPath]:
        assert (
            self.input_path_led is not None
        ), "--input-path-led cannot be None, when calling the create_led_dataset function"
        paths = list(DataPath.iterate(self.input_path_led))

        for p in paths:
            p.subject_name = p.subject_name.replace("P", "R")
            p.subject_folder = p.subject_folder.removesuffix("_LED")
            p.organ = p.organ.removesuffix("_LED")
            p.organ = self.label_renaming.get(p.organ, p.organ)

        return paths

    def copy_halogen_data(self) -> None:
        assert len(self.paths_halogen) > 0, "Could not find the single organ images"
        timestamps_original = {p.timestamp for p in self.paths_halogen}

        for timestamp in track(sorted(timestamps_original), refresh_per_second=1):
            # All folders in the single organ database with this timestamp
            paths_image = [p for p in self.paths_halogen if p.timestamp == timestamp]
            assert all(
                p.subject_name == paths_image[0].subject_name for p in paths_image
            ), f"The subject_name must be the same for all paths, error occurred on timestamp: {timestamp}"

            target_dir = self.data_dir / paths_image[0].meta("Camera_CamID") / paths_image[0].subject_name / timestamp
            target_dir.mkdir(parents=True, exist_ok=True)

            self._copy_hypergui_data(paths_image, target_dir, self.hypergui_mapping)

        self._check_hypergui_data()

    def copy_led_data(self) -> None:
        assert len(self.paths_led) > 0, "Could not find the single organ images"
        timestamps_original = {p.timestamp for p in self.paths_led}

        for timestamp in track(sorted(timestamps_original), refresh_per_second=1):
            # All folders in the single organ database with this timestamp
            paths_image = [p for p in self.paths_led if p.timestamp == timestamp]
            assert all(
                p.subject_name == paths_image[0].subject_name for p in paths_image
            ), f"The subject_name must be the same for all paths, error occurred on timestamp: {timestamp}"

            target_dir = self.data_dir / paths_image[0].meta("Camera_CamID") / paths_image[0].subject_name / timestamp
            target_dir.mkdir(parents=True, exist_ok=True)

            self._copy_hypergui_data(paths_image, target_dir, self.hypergui_mapping)

        self._check_hypergui_data()

    def copy_straylight_data(self) -> None:
        assert (
            self.input_path_straylight is not None
        ), "--input-path-straylight cannot be None, when calling the copy_straylight_data() function"
        paths = list(DataPath.iterate(self.input_path_straylight))

        straylight_mapping = {
            "org_H": "halogen",
            "wt_H": "halogen",
            "org_H_ceil": "halogen+ceiling",
            "wt_H_ceil": "halogen+ceiling",
            "org_H_ORright": "halogen+OR-right",
            "wt_H_ORright": "halogen+OR-right",
            "org_H_ORsitus": "halogen+OR-situs",
            "wt_H_ORsitus": "halogen+OR-situs",
            "org_H_ORsitus_ceil": "halogen+OR-situs+ceiling",
            "wt_H_ORsitus_ceil": "halogen+OR-situs+ceiling",
        }

        for p in track(paths, refresh_per_second=1):
            is_white = "wt_" in p.organ
            p.subject_name = p.subject_name.replace("P", "R")
            p.organ = straylight_mapping.get(p.organ, p.organ)

            if is_white:
                target_dir = self.data_dir / "straylight_experiments" / "calibration_white" / p.organ / p.timestamp
            else:
                target_dir = (
                    self.data_dir / "straylight_experiments" / "subjects" / p.organ / p.subject_name / p.timestamp
                )

            target_dir.mkdir(parents=True, exist_ok=True)
            self._copy_hypergui_data([p], target_dir, self.hypergui_mapping)

            # The straylight is not an organ label so remove the automatic image_labels creation
            meta_path = target_dir / "annotations" / f"{p.timestamp}_meta.json"
            with meta_path.open("r") as f:
                meta = json.load(f)
            meta["image_labels"].remove(p.organ)

            # Keep the information about the link to the corresponding subject
            if is_white:
                meta["calibration_subject"] = p.subject_name

            with meta_path.open("w") as f:
                json.dump(meta, f, cls=AdvancedJSONEncoder, indent=4)

        self._check_hypergui_data()

    def copy_annotations_straylight(self) -> None:
        straylight_dir = self.data_dir / "straylight_experiments" / "subjects"
        paths = list(DataPath.iterate(straylight_dir))

        for p in paths:
            subject_name = p.subject_name
            timestamp = p.timestamp
            straylight = p.straylight
            for annotator_dir in sorted(
                (settings.datasets.network_data / "annotations_raw" / self.dataset_name).iterdir()
            ):
                assert annotator_dir.is_dir(), f"{annotator_dir} is not a directory"

                nrrd_file = list(annotator_dir.glob(f"*_{timestamp}_*.nrrd"))
                assert (
                    len(nrrd_file) == 1
                ), f"Found {len(nrrd_file)} nrrd files for {straylight} {subject_name} {timestamp}"
                nrrd_file = nrrd_file[0]

                annotations_dir = straylight_dir / straylight / subject_name / timestamp / "annotations"
                annotations_dir.mkdir(parents=True, exist_ok=True)

                target_file = annotations_dir / f"{timestamp}#{annotator_dir.name}.nrrd"
                mapping = nrrd_mask(nrrd_file)["label_mapping"]
                safe_copy(nrrd_file, target_file)

                meta_labels_path = annotations_dir / f"{timestamp}_meta.json"
                if meta_labels_path.exists():
                    with meta_labels_path.open() as f:
                        meta_labels = json.load(f)
                else:
                    meta_labels = {"image_labels": []}
                meta_labels_copy = copy.deepcopy(meta_labels)

                for label in mapping.label_names():
                    if label not in meta_labels["image_labels"]:
                        meta_labels["image_labels"].append(label)

                meta_labels["image_labels"] = sort_labels(meta_labels["image_labels"])
                if meta_labels != meta_labels_copy:
                    with meta_labels_path.open("w") as f:
                        json.dump(meta_labels, f, cls=AdvancedJSONEncoder, indent=4)

    def copy_perfusion_data(self) -> None:
        assert (
            self.input_path_perfusion is not None
        ), "--input-path-perfusion cannot be None, when calling the copy_perfusion_data() function"
        paths = list(DataPath.iterate(self.input_path_perfusion, filters=[lambda p: p.attributes[1] != "Cat_reperf_baseline"]))
        paths_reperf_baseline = list(DataPath.iterate(self.input_path_perfusion, filters=[lambda p: p.attributes[1] == "Cat_reperf_baseline"]))
        assert len(paths) > 0, "Could not find the perfusion paths"
        assert len(paths_reperf_baseline) > 0, "Could not find the reperfusion baseline paths"
        assert len({p.timestamp for p in paths_reperf_baseline}) == len(paths_reperf_baseline), "Found duplicate reperfusion baseline paths"

        # Check uniqueness
        timestamps = np.asarray([p.timestamp for p in paths])
        values, counts = np.unique(timestamps, return_counts=True)
        if any(counts > 1):
            duplicate_timestamps = set(values[counts > 1])
            duplicate_paths = '\n'.join([str(p) for p in paths if p.timestamp in duplicate_timestamps])
            raise ValueError(f"The perfusion paths (without Cat_reperf_baseline) are not unique. Found multiple paths with the same timestamp:\n{duplicate_paths}")

        perfusion_type_mapping = {
            "Cat_multivisc_art_ischem": "ischemia",
            "Cat_multivisc_comb_malperf": "avascular",
            "Cat_multivisc_ven_cong": "stasis",
        }

        hypergui_mapping = {
            "_hypergui_1": {"label_name": "stomach"},
            "_hypergui_2": {"label_name": "small_bowel"},
            "_hypergui_3": {"label_name": "colon"},
            "_hypergui_4": {"label_name": "liver"},
            "_hypergui_5": {"label_name": "kidney"},
            "_hypergui_6": {"label_name": "spleen"},
            "_hypergui_7": {"label_name": "bladder"},
            "_hypergui_8": {"label_name": "omentum"},
            "_hypergui_9": {"label_name": "fat_visceral"},
        }
        for hypergui_info in hypergui_mapping.values():
            hypergui_info["annotation_type"] = "polygon"
            hypergui_info["annotator"] = "annotator1"

        copied_timestamps = set()
        for p in track(paths + paths_reperf_baseline, refresh_per_second=1):
            match = re.search(r"^P\d\d\d", p.attributes[-1])
            assert match is not None, f"Could not find the subject name for {p}"
            subject_name = match.group(0).replace("P", "R")

            target_dir = self.data_dir / "perfusion_experiments" / "subjects" / subject_name / p.timestamp
            target_dir.mkdir(parents=True, exist_ok=True)
            if p.timestamp not in copied_timestamps:
                self._copy_hypergui_data([p], target_dir, hypergui_mapping)

                meta_path = target_dir / "annotations" / f"{p.timestamp}_meta.json"
                with meta_path.open("r") as f:
                    meta = json.load(f)

                if p.attributes[1] == "Cat_baseline":
                    meta["phase_type"] = "physiological"
                    meta["phase_time"] = float("nan")
                    meta["reperfusion_baseline"] = False
                    assert "_labelling_red_perf_baseline" in meta["paper_tags"], p
                elif p.attributes[1] == "Cat_reperf_baseline":
                    meta["phase_type"] = perfusion_type_mapping[p.attributes[0]]
                    meta["phase_time"] = float("nan")
                    meta["reperfusion_baseline"] = True
                else:
                    meta["phase_type"] = perfusion_type_mapping[p.attributes[0]]
                    meta["phase_time"] = float(re.sub("Cat_(?:reperf_)?min_", "", p.attributes[1]))
                    meta["reperfusion_baseline"] = False

                if "reperf_min" in p.attributes[1]:
                    meta["reperfused"] = True
                else:
                    meta["reperfused"] = False

                with meta_path.open("w") as f:
                    json.dump(meta, f, cls=AdvancedJSONEncoder, indent=4)

                copied_timestamps.add(p.timestamp)
            else:
                assert p.attributes[1] == "Cat_reperf_baseline", f"Duplicates should only happen for reperfusion baselines: {p}"

                # Many of the reperfusion baseline images are copies from previous phase times. In these cases, we only update the reperfusion_baseline attribute
                meta_path = target_dir / "annotations" / f"{p.timestamp}_meta.json"
                with meta_path.open("r") as f:
                    meta = json.load(f)

                meta["reperfusion_baseline"] = True

                with meta_path.open("w") as f:
                    json.dump(meta, f, cls=AdvancedJSONEncoder, indent=4)

        self._check_hypergui_data()

    def copy_meta_schema(self) -> None:
        # We are using the same meta schema as the masks dataset
        safe_copy(settings.data_dirs.masks / "meta.schema", self.data_dir / "meta.schema")

    def dataset_settings(self) -> None:
        label_mapping, last_valid_label_index = self._hypergui_label_mapping()

        label_ordering_masks = DatasetSettings(settings.data_dirs.masks)["label_ordering"]
        label_ordering = {}
        old_subject_name_mapping = {}
        for p in self.paths_halogen:
            assert (
                p.organ_number == label_ordering_masks[p.organ]
            ), f"Label ordering is not in agreement with the masks dataset for {p.organ}"
            label_ordering[p.organ] = label_ordering_masks[p.organ]
            # assert same label ordering as in masks
            old_subject_name_mapping[p.subject_name] = p.subject_folder

        for p in self.paths_led:
            assert p.organ in label_ordering, f"The organ {p.organ} is not part of the halogen dataset"
            assert label_ordering[p.organ] == p.organ_number, (
                f"The organ number for {p.organ} is not the same in the halogen and led dataset:"
                f" {label_ordering[p.organ] = } != {p.organ_number = }"
            )
            assert (
                p.subject_name in old_subject_name_mapping
            ), f"The subject {p.subject_name} is not part of the halogen dataset"
            assert old_subject_name_mapping[p.subject_name] == p.subject_folder, (
                f"The subject folder for {p.subject_name} is not the same in the halogen and led dataset"
                f" {old_subject_name_mapping[p.subject_name] = } != {p.subject_folder = }"
            )

        # For the additional straylight labels, use the same ordering as in the masks dataset
        for label, index in label_mapping.items():
            if index <= last_valid_label_index and label not in label_ordering.keys():
                label_ordering[label] = label_ordering_masks[label]

        # Sort by values (organ numbers)
        label_ordering = dict(sorted(label_ordering.items(), key=lambda item: item[1]))

        # Sort by keys (subject names)
        old_subject_name_mapping = dict(sorted(old_subject_name_mapping.items()))

        dataset_settings = {
            "dataset_name": "2023_12_07_Tivita_multiorgan_rat",
            "data_path_class": "htc.tivita.DataPathMultiorganCamera>DataPathMultiorganCamera",
            "shape": [480, 640, 100],
            "shape_names": ["height", "width", "channels"],
            "label_mapping": label_mapping,
            "last_valid_label_index": last_valid_label_index,
            "annotation_name_default": "polygon#annotator1",
            "annotator_mapping": {
                "annotator1": "Lotta Biehl",
                "annotator2": "Lotta Biehl",
                "annotator3": "Silvia Seidlitz",
            },
            "camera_type": {
                "0202-00118": "Halogen",
                "0615-00036": "LED",
            },
            "label_ordering": label_ordering,
            "old_subject_name_mapping": old_subject_name_mapping,
            "sex": {
                "R002": "f",
                "R003": "f",
                "R014": "f",
                "R015": "f",
                "R016": "f",
                "R017": "f",
                "R018": "f",
                "R019": "f",
                "R020": "f",
                "R021": "f",
                "R022": "f",
                "R023": "f",
                "R024": "f",
                "R025": "f",
                "R026": "f",
                "R027": "m",
                "R028": "m",
                "R029": "m",
                "R030": "m",
                "R031": "m",
                "R032": "m",
                "R033": "m",
                "R034": "m",
                "R035": "m",
                "R036": "m",
            },
        }

        with (self.data_dir / "dataset_settings.json").open("w") as f:
            json.dump(dataset_settings, f, cls=AdvancedJSONEncoder, indent=4, ensure_ascii=False)

        dataset_settings["data_path_class"] = "htc.tivita.DataPathMultiorganStraylight>DataPathMultiorganStraylight"
        dataset_settings["annotation_name_default"] = "polygon#annotator1"
        with (self.data_dir / "straylight_experiments" / "dataset_settings.json").open("w") as f:
            json.dump(dataset_settings, f, cls=AdvancedJSONEncoder, indent=4, ensure_ascii=False)

        dataset_settings["data_path_class"] = "htc.tivita.DataPathMultiorgan>DataPathMultiorgan"
        dataset_settings["annotation_name_default"] = "polygon#annotator1"
        with (self.data_dir / "perfusion_experiments" / "dataset_settings.json").open("w") as f:
            json.dump(dataset_settings, f, cls=AdvancedJSONEncoder, indent=4, ensure_ascii=False)

    def recalibration_match(self, path: DataPath) -> Union[DataPath, None]:
        """
        This function matches an image to the corresponding white tile image for recalibration.
        """
        if (
            path not in self.dataset_paths["straylight_experiments"]
        ):  # Only recalibrate straylight images for now as other images should be well calibrated.
            return None
        elif path.subject_name == "calibration_white":  # Don't recalibrate white tile images.
            return None

        white_paths = list(DataPath.iterate(settings.data_dirs.rat / "straylight_experiments" / "calibration_white"))
        white_paths = [
            p
            for p in white_paths
            if p.straylight == path.straylight and p.meta("calibration_subject") == path.subject_name
        ]

        if len(white_paths) == 0:
            settings.log.error(f"No white tile image with matching subject_name and date found for {path.image_name()}")
            return None
        elif len(white_paths) > 1:
            settings.log.info(
                f"Multiple white tile images with matching subject_name and date found for {path.image_name()}. The"
                " one closest in time is used."
            )
            white_paths = sorted(white_paths, key=lambda p: abs(p.datetime() - path.datetime()))

        return recalibrated_path(path, white_paths[0])

    @classmethod
    def from_parser(cls, **kwargs) -> Self:
        if "additional_arguments" not in kwargs:
            kwargs["additional_arguments"] = {}

        kwargs["additional_arguments"] |= {
            "--input-path-halogen": {
                "type": Path,
                "required": False,
                "default": None,
                "help": (
                    "Path to the Cat_atlas folder with the data from the halogen camera (It should contain folders with"
                    " the name Cat_*)."
                ),
            },
            "--input-path-led": {
                "type": Path,
                "required": False,
                "default": None,
                "help": (
                    "Path to the Cat_atlas_LED folder with the data from the led camera (It should contain folders with"
                    " the name Cat_*LED)."
                ),
            },
            "--input-path-straylight": {
                "type": Path,
                "required": False,
                "default": None,
                "help": (
                    "Path to the Cat_light folder with the data from the straylight experiments (It should contain"
                    " folders with the names Cat_0001_org_H, Cat_0001_wt_H etc.)"
                ),
            },
            "--input-path-perfusion": {
                "type": Path,
                "required": False,
                "default": None,
                "help": (
                    "Path to the folder which contains the perfusion rats data (Cat_multivisc_art_ischem,"
                    " Cat_multivisc_comb_malperf and Cat_multivisc_ven_cong folders)"
                ),
            },
        }

        return super().from_parser(**kwargs)


if __name__ == "__main__":
    # screen -S dataset_rat -d -m script -q -c "htc dataset_rat --input-path-halogen /mnt/hdd_36tb/Cat_Rat/Cat_atlas --input-path-led /mnt/hdd_36tb/Cat_Rat_LED/Cat_atlas_LED --input-path-straylight /mnt/hdd_36tb/Cat_light --input-path-perfusion /mnt/hdd_36tb/Cat_perfusion" dataset_rat.log
    dataset_path = settings.datasets.rat["path_dataset"]
    generator = DatasetGeneratorRat.from_parser(output_path=dataset_path)
    generator.copy_halogen_data()
    generator.copy_led_data()
    generator.copy_straylight_data()
    generator.copy_annotations_straylight()
    generator.copy_perfusion_data()
    generator.copy_meta_schema()
    generator.run_safe(generator.dataset_settings)
    p_map(generator.segmentations, generator.paths, num_cpus=2.0, task_name="Segmentation files")
    generator.meta_table()
    generator.median_spectra_table()
    generator.median_spectra_table(recalibration_match=generator.recalibration_match)
    generator.median_spectra_table(oversaturation_threshold=1000)
    generator.median_spectra_table(recalibration_match=generator.recalibration_match, oversaturation_threshold=1000)
    generator.preprocessed_files()
    generator.view_organs()
