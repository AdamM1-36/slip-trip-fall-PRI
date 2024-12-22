#!/usr/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pri3.12
uvicorn integrated:app --host 0.0.0.0 --port 5000 --reload
