#!/bin/bash
source activate /home/l328r/.conda/envs/htc-dev
input_paths=(
    "/omics/groups/OE0645/internal/data/htcdata/medium_test/Cat_uro_conduit__base"
    
)


python  /home/l328r/htc/htc/dataset_preparation/run_dataset_alex.py --input-path "${input_paths[@]}" --regenerate --include-csv