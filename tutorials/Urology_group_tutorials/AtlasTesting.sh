#!/bin/bash
#sh script to run training of an htc model. The hard work of building the config json should aready be done in a jupyter notebook, and simply be referencede here
#aslo the path env variables to external and results shopuld also be set


source /home/l328r/.bashrc
eval "$(conda shell.bash hook)"
conda activate htc-dev

# Define the path to your configuration file as a string
run_folder="/omics/groups/OE0645/internal/data/htcdata/medium_test/results/training/median_pixel/2024-07-25_17-43-32_Atlas_config"

htc median_test_table --model median_pixel --run-folder $run_folder --spec $run_folder/"data.json"  --table-name test_table

#After saving, before submitting job, do: 
#chmod +x </path/to/>training.sh
#bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu //home/l328r/htc/tutorials/Atlas_tutorials/training.sh