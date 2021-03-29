#!/bin/bash

E_NOARGS=65
E_NODIR=67

if [ -z "$1" ]
then
  echo "Usage: `basename $0` data_dir out_dir"
  exit $E_NOARGS
else
    if [ -d "$1" ]
    then
        data_dir="${1%/}/"
    fi
fi

if [ -n "$2" ]
then
    out_dir="${2%/}/"
    if [ ! -d "$2" ]
    then
        # Create output directory
        mkdir -p $out_dir
    fi
else
    echo "Output directory not specified"
    echo "Usage: $0 data_dir out_dir"
    exit $E_NODIR
fi

__conda_setup="$($CONDA_EXE 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    conda_path=$(dirname $(dirname $CONDA_EXE))
    if [ -f "${conda_path}/etc/profile.d/conda.sh" ]; then
        . "${conda_path}/etc/profile.d/conda.sh"
    else
        export PATH="${conda_path}/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate notebook

python python/tma_alert_check.py $data_dir $out_dir

conda deactivate
