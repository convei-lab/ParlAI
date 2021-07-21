#!/bin/zsh
# SPECIFY YOUR SHELL PROGRAM IN THE ABOVE SHABANG!
# YOU NEED TO RUN THIS SHELL SCRIPT IN [path-to-parlai]/ParlAI/scripts/
# FOR SUCCESSFUL ENVIRONMENT
# ALSO MAKE SURE YOU PROPERLY INSTALL CONDA AND INIT THEM
envname='blending-test'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/common/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/common/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/common/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/common/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
set -o xtrace
conda create -n $envname python=3.7 -y
conda activate $envname 
cd ..
yes | pip install -e .
# YOU NEED TO MAKE SURE THE CUDA TOOLKIT VERSION HERE
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

# NOW YOU SOURCE YOUR ENVIRONMENT AND RUN ANY SCRIPT
# e.g. conda activate [your-env-name]
#      git checkout -b [your-git-branch-name]
