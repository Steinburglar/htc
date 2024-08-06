# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.utils.LabelMapping import LabelMapping
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path


class SettingsUreter:
    def __init__(self):
        self.label_mapping = LabelMapping({
        "last_valid_label_index": 2,
        "mapping_index_name": {
            "0": "background_anorganic",
            "1": "background_organic",
            "2": "ureter",
            },
        "mapping_name_index": {
            "background_anorganic": 0, 
            "background_organic": 1,
            "ureter_left": 2,
            "ureter_right": 2,
            },
            })
        self.labels = self.label_mapping.label_names()
        self.figure_labels = [k.replace("_", " ") for k in self.labels]


        
        self.label_colors = {
            {"ureter": "#a7e89b"},
            {"background_organic": "#a86d32"},
            {"background_anorganic": "#ED00D2"}
            }

        self._results_dir = None
