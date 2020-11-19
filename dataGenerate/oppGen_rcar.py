import argparse
import os
import re
import matplotlib.pyplot as pyt
from datetime import datetime as dt
from datetime import timedelta
from image import load_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Get opp image pairs')

parser.add_argument('--dir', type=str, default='/scratch/dhagash/robotcar/2014-05-06-12-54-54/', help='Parent Directory.')

args = parser.parse_args()

timestamps_path_stereo = os.path.join(args.dir, 'stereo.timestamps')
timestamps_path_mono = os.path.join(args.dir, 'mono_rear.timestamps')

if not os.path.isfile(timestamps_path_stereo or timestamps_path_mono):
  raise IOError("Could not find timestamps file")

timestamps_file_front = open(timestamps_path_stereo)
timestamps_file_rear = open(timestamps_path_mono)
front = []
rear = []
pair = []
for line_f, line_r in zip(timestamps_file_front, timestamps_file_rear):
    front.append(int(line_f.split()[0]))
    rear.append(int(line_r.split()[0]))
k = 50
for i in range(len(rear) - k):
    if rear[i + k] in front:
        print('brb', front[i], rear[i+k])
    else:
        print('whew', front[min(range(len(front)), key=lambda j:abs(front[j] - rear[i+k]))-k], rear[i+k])
