# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Union

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings


class DataPathTivita(DataPath):
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructs a generic data path for any kind of Tivita hyperspectral image folder.
        """
        super().__init__(*args, **kwargs)
        self.attributes = list(self.image_dir.relative_to(self.data_dir).parts[:-1])

    def build_path(self, base_folder: Path) -> Path:
        return base_folder / "/".join(self.attributes + [self.timestamp])

    @staticmethod
    def from_image_name(image_name: str) -> "DataPathTivita":
        raise NotImplementedError()

    @staticmethod
    def iterate(
        data_dir: Path,
        filters: list[Callable[["DataPath"], bool]],
        annotation_name: Union[str, list[str]],
    ) -> Iterator["DataPathTivita"]:
        # Settings of the dataset (shapes etc.) can be referenced by the DataPaths
        path_settings = None
        possible_paths = [data_dir] + list(data_dir.parents)
        for p in possible_paths:
            if (p / "dataset_settings.json").exists():
                path_settings = p / "dataset_settings.json"
                break

        dataset_settings = DatasetSettings(path_settings)
        intermediates_dir = settings.data_dirs.find_intermediates_dir(data_dir)

        parents = {}
        for p in sorted(data_dir.rglob("*")):
            if p.name.endswith("SpecCube.dat") or p.suffix == ".tiv":
                parents[p.parent] = True

        for p in parents.keys():
            path = DataPathTivita(p, data_dir, intermediates_dir, dataset_settings, annotation_name)
            if all([f(path) for f in filters]):
                yield path