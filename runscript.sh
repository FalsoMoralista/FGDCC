
#!/bin/bash

source ~/miniconda3/bin/activate py382

sbatch -J exp_52 --gpus 1 --mincpus 16 FGDCC_v2.sh 
