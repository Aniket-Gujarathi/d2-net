import argparse
import os
import pandas as pd

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
k = 55
# for i in range(len(rear) - k):
#     if rear[i + k] in front:
#         print('brb', front[i], rear[i+k])
#     else:
#         print('whew', front[min(range(len(front)), key=lambda j:abs(front[j] - rear[i+k]))-k], rear[i+k])
front_pair = []
rear_pair = []
for i in range(len(front) - k):
    if front[i + k] in rear:
        front_pair.append(front[i])
        rear_pair.append(rear[i + k])
        print('y', front[i], rear[i + k])
    else:
        rear_closest = rear[min(range(len(rear)), key=lambda j:abs(rear[j] - front[i + k]))]
        front_pair.append('/scratch/dhagash/robotcar/2014-05-06-12-54-54/stereo/centre/' + str(front[i]) + '.png')
        rear_pair.append('/scratch/dhagash/robotcar/2014-05-06-12-54-54/mono_rear/' + str(rear_closest) + '.png')

        if rear_closest == 1399381573789306:
            break

        print('n', front[i], rear_closest)

df = pd.DataFrame({'front' :  front_pair, 'rear' : rear_pair})
df.to_csv('/home/dhagash/d2-net/dataGenerate/rcar_oppPairs.csv', index=False)
