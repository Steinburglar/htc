#!/bin/bash
#sh script to run training of an htc model. The hard work of building the config json should aready be done in a jupyter notebook, and simply be referencede here
#aslo the path env variables to external and results shopuld also be set


source /home/l328r/.bashrc
eval "$(conda shell.bash hook)"
conda activate htc-dev

# Define the path to your configuration file as a string
config_path="/omics/groups/OE0645/internal/data/htcdata/medium_test/external/data/Atlas_config.json"

htc training --model image --config $config_path #make sure the model matches the model you want to use (i.e, image for segmetation, or median_pixel for tissue spectra classification)

#After saving, before submitting job, do: 
#chmod +x training.sh
#bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu //home/l328r/htc/tutorials/Atlas_tutorials/training.sh