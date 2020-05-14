#!/bin/bash

cd ./vaw-gan-f0

python convert-vawgan-separate-f0.py --checkpoint "./log/train/MODEL_NAME/model.ckpt-46860"
python save_f0.py

cd ../vaw-gan

python test-condition.py
python convert-vawgan-cwt.py --checkpoint "./log/train/MODEL_NAME/model.ckpt-46860"



