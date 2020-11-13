#!/bin/bash

cd ./vaw-gan-f0

python convert-vawgan-separate-f0.py #Please customize your checkpoint dir in python script: --checkpoint "./log/train/[TIMESETP]/model.ckpt-[MODEL ID]"
python convert-vawgan-separate-f0.py #Please customize your checkpoint dir in python script: --checkpoint "./log/train/[TIMESETP]/model.ckpt-[MODEL ID]"
python save_f0.py

cd ../vaw-gan

python test-condition.py
python convert-vawgan-cwt.py #Please customize your checkpoint dir in python script: --checkpoint "./log/train/[TIMESETP]/model.ckpt-[MODEL ID]"
python convert-vawgan-cwt.py #Please customize your checkpoint dir in python script: --checkpoint "./log/train/[TIMESETP]/model.ckpt-[MODEL ID]"



