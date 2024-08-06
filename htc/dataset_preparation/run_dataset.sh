#!/bin/bash
source activate /home/l328r/.conda/envs/htc-dev
input_paths=(
    "/omics/groups/OE0645/internal/data/htcdata/ureter/ureter_images" 
)


python  /home/l328r/htc/htc/dataset_preparation/run_dataset_ureter.py --input-path "${input_paths[@]}" --regenerate --include-csv