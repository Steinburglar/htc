import json
from functools import cached_property, partial
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
from PIL import Image
from rich.progress import track
from typing_extensions import Self

from htc.dataset_preparation.DatasetGenerator import DatasetGenerator
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.blosc_compression import compress_file
from htc.utils.helper_functions import sort_labels
from htc.utils.LabelMapping import LabelMapping
from htc.utils.mitk.mitk_masks import nrrd_mask
from htc.utils.parallel import p_map
from htc.utils.visualization import compress_html, create_overview_document


class DatasetGeneratorMasks(DatasetGenerator):
    def __init__(self, input_path_open_data: Path = None, **kwargs):
        """
        Dataset generation of the mask dataset.

        Args:
            input_path_open_data: Path to the open dataset from the clinicians (unstructured, it should contain folders with the name `Cat_*`).
        """
        super().__init__(**kwargs)
        self.input_path_open_data = input_path_open_data
        if self.input_path_open_data is not None:
            assert self.input_path_open_data.is_dir(), "The input_path_open_data does not point towards a directory."

        # There may be more than one hypergui folder
        if "2021_02_05_Tivita_multiorgan_masks" in self.dataset_name:
            n_hypergui_folder = 3
        else:
            n_hypergui_folder = 1

        self.hypergui_mapping = {}
        for r in range(1, n_hypergui_folder + 1):
            self.hypergui_mapping[f"_hypergui_{r}"] = {
                "annotation_type": "polygon",
                "annotator": f"annotator{r}",
            }

        self.overview_kwargs = {}

    @cached_property
    def paths_halogen(self) -> list[DataPath]:
        assert self.input_path is not None, "--input-path cannot be None, when calling the create_dataset function"
        paths = list(DataPath.iterate(self.input_path))

        label_renaming = {
            "abdominal_linen_total": "abdominal_linen",
            "kidney_with_peritoneum": "kidney_with_Gerotas_fascia",
            "white_compress_total": "white_compress",
            "metal_parts": "metal",
        }

        for p in paths:
            p.organ = label_renaming.get(p.organ, p.organ)

        return paths

    def create_dataset(self) -> None:
        assert len(self.paths_halogen) > 0, "Could not find the single organ images"
        timestamps_original = {p.timestamp for p in self.paths_halogen}

        if "2021_02_05_Tivita_multiorgan_masks" in self.dataset_name:
            paths_semantic = list(DataPath.iterate(settings.data_dirs.semantic))
            timestamps_overlap = {p.timestamp for p in paths_semantic}
            timestamps_overlap = timestamps_overlap.intersection(timestamps_original)
        else:
            timestamps_overlap = set()

        for timestamp in track(sorted(timestamps_original), refresh_per_second=1):
            # All folders in the single organ database with this timestamp
            paths_image = [p for p in self.paths_halogen if p.timestamp == timestamp]
            assert all(
                p.subject_name == paths_image[0].subject_name for p in paths_image
            ), f"The subject_name must be the same for all paths, error occurred on timestamp: {timestamp}"

            if timestamp in timestamps_overlap:
                target_dir = self.data_dir / "overlap" / "subjects" / paths_image[0].subject_name / timestamp
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_dir = self.data_dir / "subjects" / paths_image[0].subject_name / timestamp
                target_dir.mkdir(parents=True, exist_ok=True)

            self._copy_hypergui_data(paths_image, target_dir, self.hypergui_mapping)

        self._check_hypergui_data()

    def fix_labelling_files(self) -> None:
        # Unfortunately, the _labelling_standardized_final label is wrong in the tissue atlas so we have to manually set the label based on the open data paper
        assert (
            self.dataset_name == "2021_02_05_Tivita_multiorgan_masks"
        ), "This function is only used for the tissue atlas 2021_02_05_Tivita_multiorgan_masks dataset"
        assert (
            self.input_path_open_data is not None
        ), "output-path-open-data argument should not be None while creating 2021_02_05_Tivita_multiorgan_masks dataset"
        paths_data = list(DataPath.iterate(self.input_path_open_data))
        timestamps_data = [p.timestamp for p in paths_data]
        paths = list(DataPath.iterate(settings.data_dirs.masks))
        paths += list(DataPath.iterate(settings.data_dirs.masks / "overlap"))

        for p in track(paths, refresh_per_second=1):
            meta = p.read_annotation_meta()
            if "paper_tags" in meta:
                if "_labelling_standardized_final" in meta["paper_tags"]:
                    meta["paper_tags"].remove("_labelling_standardized_final")

                if p.timestamp in timestamps_data:
                    meta["paper_tags"].append("_labelling_standardized_final")

                meta["paper_tags"] = sorted(meta["paper_tags"])

            if "image_labels" in meta:
                meta["image_labels"] = sort_labels(meta["image_labels"])

            with p.annotation_meta_path().open("w") as f:
                json.dump(meta, f, cls=AdvancedJSONEncoder, indent=4)

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
            # for the 2021_02_05_Tivita_multiorgan_masks dataset: Pigs 77 (including) to 84 (including) used the wrong yellow filter
            "camera_name_changes": {
                "0102-00098": {
                    "change_date": "2021_03_31_15_54_17",
                    "suffix_before": "wrong-1",
                    "suffix_after": "correct-1",
                },
                "0202-00118": {
                    "change_date": "2021_03_31_15_54_17",
                    "suffix_before": "wrong-1",
                    "suffix_after": "correct-1",
                },
            },
            "label_ordering": {
                "stomach": "0001",
                "small_bowel": "0002",
                "colon": "0003",
                "liver": "0004",
                "gallbladder": "0005",
                "pancreas": "0006",
                "ureter": "0007.1",
                "kidney": "0007",
                "spleen": "0008",
                "bladder": "0009",
                "omentum": "0010",
                "esophagus": "0012",
                "pleura": "0013.1",
                "trachea": "0013.2",
                "thyroid": "0013.3",
                "saliv_gland": "0013.4",
                "teeth": "0013.5",
                "lung": "0013",
                "heart": "0014",
                "cartilage": "0015",
                "tendon": "0016.1",
                "skel_muscle": "0016.2",
                "ligament_pat": "0016.3",
                "bone": "0016",
                "fur": "0017.1",
                "skin": "0017",
                "fat_subcutaneous": "0018.1",
                "muscle": "0018",
                "peritoneum": "0019",
                "aorta": "0020",
                "arteries": "0021",
                "major_vein": "0022",
                "veins": "0023",
                "kidney_with_Gerotas_fascia": "0024",
                "lymph_nodes": "0025",
                "diaphragm": "0026",
                "tube": "0027.1",
                "ovary": "0027.2",
                "vesic_gland": "0027.3",
                "fat_visceral": "0028",
                "thymus": "0029",
                "multiorgan_visceral": "0030",
                "multiorgan_thoracic": "0031",
                "multiorgan_visceral_retro": "0032",
                "blue_cloth": "0040",
                "white_compress": "0041",
                "white_compress_dry": "0042",
                "white_compress_wet": "0043",
                "abdominal_linen": "0044",
                "abdominal_linen_dry": "0045",
                "abdominal_linen_wet": "0046",
                "silicone_pad_white": "0050",
                "silicone_pad_yellow": "0051",
                "silicone_pad_blue": "0052",
                "silicone_pad_red": "0053",
                "silicone_gloves_white": "0055",
                "silicone_gloves_blue": "0056",
                "silicone_gloves_black": "0057",
                "silicone_gloves_light_blue": "0058",
                "metal": "0060",
                "stapler": "0061",
                "magnets": "0062",
                "sutures": "0063",
                "flowmeter": "0064",
                "other_background": "0070",
                "other_stuff": "0080",
                "blood": "0081",
                "bile_fluid": "0082",
                "lymph_fluid": "0083",
                "urine": "0084",
                "unclear_organic": "9997",
                "background": "9998",
                "unsure": "9999",
            },
            "old_subject_name_mapping": {
                "P001": "P001_OP001_2018_07_26_Experiment1",
                "P002": "P002_OP002_2018_08_06_Experiment1",
                "P003": "P003_OP003_2018_08_15_Experiment1",
                "P005": "P005_OP004_2018_09_18_Experiment1",
                "P006": "P006_OP005_2018_10_04_Experiment1",
                "P007": "P007_OP006_2018_10_04_Experiment2",
                "P008": "P008_OP007_2018_10_08_Experiment1",
                "P009": "P009_OP008_2018_10_08_Experiment2",
                "P010": "P010_OP009_2018_10_29_Experiment1",
                "P011": "P011_OP010_2018_10_29_Experiment2",
                "P012": "P012_OP011_2018_11_08_Experiment1",
                "P013": "P013_OP012_2018_11_08_Experiment2",
                "P014": "P014_OP013_2018_11_11_Experiment1",
                "P015": "P015_OP014_2018_11_11_Experiment2",
                "P016": "P016_OP015_2018_11_12_Experiment1",
                "P017": "P017_OP016_2018_11_15_Experiment1",
                "P018": "P018_OP000_2018_12_01_Subexperiment1",
                "P019": "P019_OP017_2019_03_16_Experiment1",
                "P020": "P020_OP018_2019_03_16_Experiment2",
                "P021": "P021_OP019_2019_03_17_Experiment1",
                "P022": "P022_OP020_2019_03_31_Experiment1",
                "P023": "P023_OP021_2019_03_31_Experiment2",
                "P024": "P024_OP022_2019_04_13_Experiment1",
                "P025": "P025_OP023_2019_04_15_Experiment1",
                "P026": "P026_OP024_2019_04_23_Experiment1",
                "P028": "P028_OP026_2019_05_18_Experiment1",
                "P029": "P029_OP027_2019_06_08_Experiment1",
                "P030": "P030_OP028_2019_06_20_Experiment1",
                "P031": "P031_OP000_2019_09_16_Subexperiment1",
                "P033": "P033_OP000_2019_10_16_Subexperiment1",
                "P034": "P034_OP000_2019_11_11_Subexperiment1",
                "P036": "P036_OP030_2019_11_26_Experiment1",
                "P037": "P037_OP031_2019_11_27_Experiment1",
                "P039": "P039_OP000_2019_11_30_Experiment1",
                "P040": "P040_OP000_2019_11_30_Experiment1",
                "P041": "P041_Pig001_Pilot_2019_12_14_Experiment1",
                "P042": "P042_Pig002_G001_2019_12_15_Experiment1",
                "P043": "P043_OP033_2019_12_20_Experiment1",
                "P044": "P044_Pig003_G001_2020_02_01",
                "P045": "P045_Pig004_G001_2020_02_05",
                "P046": "P046_Pig005_G001_2020_02_07",
                "P047": "P047_Pig006_G001_2020_02_07",
                "P048": "P048_Pig007_G001_2020_02_08",
                "P049": "P049_Pig008_G002_2020_02_11",
                "P050": "P050_Pig009_G002_2020_02_18",
                "P051": "P051_Pig010_G002_2020_03_03",
                "P052": "P052_Pig011_G002_2020_03_04",
                "P053": "P053_Pig012_G002_2020_03_06",
                "P054": "P054_Pig013_G002_2020_03_10",
                "P055": "P055_Pig014_G004_2020_03_11",
                "P056": "P056_Pig000_G004_2020_03_12",
                "P057": "P057_Pig015_G004_2020_03_13",
                "P058": "P058_OP034_2020_05_13_Experiment1",
                "P059": "P059_Pig016_G004_2020_05_14",
                "P060": "P060_OP035_2020_05_14_Experiment2",
                "P061": "P061_Pig017_G004_2020_05_15",
                "P062": "P062_OP036_2020_05_15_Experiment2",
                "P063": "P063_Pig018_G004_2020_05_28",
                "P064": "P064_Pig019_G004_2020_05_29",
                "P065": "P065_Pig020_G004_2020_06_19",
                "P066": "P066_Pig021_G004_2020_07_07",
                "P067": "P067_Pig022_G002_2020_07_09",
                "P068": "P068_Pig023_G004_2020_07_20",
                "P069": "P069_Pig024_G004_2020_07_23",
                "P070": "P070_Pig025_G001_2020_07_24",
                "P071": "P071_Pig026_G001_2020_08_05",
                "P072": "P072_Pig027_G002_2020_08_08",
                "P074": "P074_OP038_2020_08_19_Experiment1",
                "P075": "P075_OP039_2020_08_20_Experiment1",
                "P076": "P076_OP040_2020_08_24_Experiment1",
                "P077": "P077_OP041_2021_01_31_Experiment1",
                "P078": "P078_OP042_2021_02_07_Experiment1",
                "P079": "P079_OP043_2021_02_11_Experiment1",
                "P080": "P080_OP044_2021_02_14_Experiment1",
                "P081": "P081_OP045_2021_02_23_Experiment1",
                "P082": "P082_OP046_2021_02_25_Experiment1",
                "P083": "P083_OP047_2021_03_05_Experiment1",
                "P084": "P084_OP048_2021_03_21_Experiment1",
                "P085": "P085_OP049_2021_04_10_Experiment1",
                "P086": "P086_OP050_2021_04_15_Experiment1",
                "P087": "P087_OP051_2021_04_16_Experiment1",
                "P088": "P088_OP052_2021_04_19_Experiment1",
                "P089": "P089_OP053_2021_04_21_Experiment1",
                "P090": "P090_OP054_2021_04_22_Experiment1",
                "P091": "P091_OP055_2021_04_24_Experiment1",
                "P092": "P092_OP056_2021_04_27_Experiment1",
                "P093": "P093_OP057_2021_04_28_Experiment1",
                "P094": "P094_OP058_2021_04_30_Experiment1",
                "P095": "P095_OP059_2021_05_02_Experiment1",
                "P096": "P096_OP060_2021_05_06_Experiment1",
                "P097": "P097_OP061_2021_07_03_Experiment1",
                "P098": "P098_OP062_2021_07_06_Experiment1",
                "P099": "P099_OP063_2021_07_18_Experiment1",
                "P100": "P100_OP064_2021_08_04_Experiment1",
            },
            "sex": {
                "P086": "m",
                "P087": "f",
                "P088": "f",
                "P089": "m",
                "P090": "f",
                "P091": "f",
                "P092": "f",
                "P093": "f",
                "P094": "m",
                "P095": "m",
                "P096": "m",
            },
        }

        with (self.data_dir / "overlap" / "dataset_settings.json").open("w") as f:
            json.dump(dataset_settings, f, cls=AdvancedJSONEncoder, indent=4, ensure_ascii=False)

        with (self.data_dir / "dataset_settings.json").open("w") as f:
            json.dump(dataset_settings, f, cls=AdvancedJSONEncoder, indent=4, ensure_ascii=False)

    def segmentations(self, path: DataPath) -> None:
        label_mapping = LabelMapping.from_data_dir(self.data_dir)
        target_dir = self.intermediates_dir / "segmentations"
        target_dir.mkdir(parents=True, exist_ok=True)

        annotations_dir = path() / "annotations"
        if not annotations_dir.exists():
            return None

        annotations = {}
        for f in sorted(annotations_dir.iterdir()):
            if f.suffix == ".png":
                timestamp, annotation_type, annotator, label_name, file_type = f.stem.split("#")
                assert label_name in label_mapping, f"No label_index defined for the label {label_name}"
                assert timestamp == path.timestamp
                assert file_type == "binary"

                # Read mask
                mask = np.array(Image.open(f))
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

        if len(annotations) > 0:
            compress_file(target_dir / f"{path.image_name()}.blosc", annotations)

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

    def _save_html(self, path: DataPath, navigation_paths: list[DataPath]) -> None:
        def create_navigation_link(label_name: str, label_order: str, image_path: DataPath) -> str:
            return f"../{label_order}_{label_name}/{quote_plus(image_path.image_name())}.html"

        html = create_overview_document(
            path,
            navigation_paths=navigation_paths,
            navigation_link_callback=create_navigation_link,
            **self.overview_kwargs,
        )
        html = compress_html(file=None, fig_or_html=html)

        for label_name in path.annotated_labels(annotation_name="all"):
            target_dir = (
                self.intermediates_dir / "view_organs" / f"{self.dsettings['label_ordering'][label_name]}_{label_name}"
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            (target_dir / f"{path.image_name()}.html").write_text(html)

    @classmethod
    def from_parser(cls, **kwargs) -> Self:
        if "additional_arguments" not in kwargs:
            kwargs["additional_arguments"] = {}

        kwargs["additional_arguments"] |= {
            "--input-path-open-data": {
                "type": Path,
                "required": False,
                "default": None,
                "help": "Path to the open dataset (It should contain folders with the name Cat_*).",
            },
        }

        return super().from_parser(**kwargs)


if __name__ == "__main__":
    # The path argument for DatasetGenerator can be set to 2021_02_05_Tivita_multiorgan_masks dataset paths. Example:
    # screen -S dataset_masks -d -m script -q -c "htc dataset_masks --input-path /mnt/E130-Projekte/Biophotonics/Data/2020_07_23_hyperspectral_MIC_organ_database/data/Catalogization_tissue_atlas" dataset_masks.log
    dataset_path = settings.datasets.masks["path_dataset"]
    generator = DatasetGeneratorMasks.from_parser(output_path=dataset_path)

    generator.create_dataset()

    if "2021_02_05_Tivita_multiorgan_masks" in generator.dataset_name:
        generator.fix_labelling_files()

    generator.run_safe(generator.dataset_settings)

    p_map(generator.segmentations, generator.paths, num_cpus=2.0, task_name="Segmentation files")
    generator.meta_table()
    generator.median_spectra_table()
    generator.preprocessed_files()
    generator.view_organs()
