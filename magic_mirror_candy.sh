#!/bin/sh

python stylize_webcam.py --capture_device 1 --vertical --fullscreen --canvas_size 540 960 --model_path models/candy_final.ckpt $*
