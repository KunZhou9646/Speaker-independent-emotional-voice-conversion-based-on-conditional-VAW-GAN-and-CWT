#!/bin/bash

cd ./vaw-gan-f0

python analyzer_f0.py
python build_f0.py
python main-vawgan-f0.py


