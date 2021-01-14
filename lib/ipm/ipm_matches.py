from PIL import Image
import cv2
import numpy as np
import pandas as pd
from ipm import ipm
import random

def getimgPair(imgData):
    imgFiles = []
    data = pd.read_csv(imgData, sep=',', header=1, names=['front', 'rear'])
    data = data.sample(n=100)
    imgFiles = [data['front'], data['rear']]

    return imgFiles

def build_dataset(imgFiles):
    cfgFront = '/home/udit/d2-net/lib/ipm/cameraFront.json'
    cfgRear = '/home/udit/d2-net/lib/ipm/cameraRear.json'
    for i in range(len(imgFiles[0][:])):

        front_top = ipm(imgFiles[0][i], cfgFront)
        rear_top = ipm(imgFiles[1][i], cfgRear)
        front_top = Image.fromarray(front_top).convert('LA').convert('RGB')
        rear_top = Image.fromarray(rear_top).convert('LA').convert('RGB')

        front_top.save('/scratch/udit/robotcar/overcast/ipm/front/' + str(i) + '.png')
        rear_top.save('/scratch/udit/robotcar/overcast/ipm/rear/' + str(i) + '.png')
        print(i)

if __name__ == "__main__":
    imgData = '/home/udit/d2-net/dataGenerate/imagePairsOxford.csv'
    imgFiles = np.array(getimgPair(imgData))
    #print(imgFiles[0][0], imgFiles[1][0])
    build_dataset(imgFiles)
