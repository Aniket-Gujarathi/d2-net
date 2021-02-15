import numpy
from PIL import Image
import cv2
from sys import argv, exit
import matplotlib.pyplot as plt
import os

def getTop(front_road, rear_road, front_H, rear_H):
    imgSize = 800

    ftop = cv2.warpPerspective(front_road, front_H, (imgSize, imgSize))
    rtop = cv2.warpPerspective(rear_road, rear_H, (imgSize, imgSize))

    return ftop, rtop

def getRoad(img, segm):
    for u in range(img.shape[0]):
        for v in range(img.shape[1]):
            if segm.getpixel((u, v)) == 255:
                continue
            else:
                img.putpixel((u, v), (0, 0, 0))

    return img

if __name__ == '__main__':
    frontRGBFolder = argv[1]
    rearRGBFolder = argv[2]
    frontSegFolder = argv[3]
    rearSegFolder = argv[4]

    front_H = np.load('front_H.npy')
    rear_H = np.load('rear_H.npy')

    for front, rear, fseg, rseg in zip(os.listdir(frontRGBFolder), os.listdir(rearRGBFolder), os.listdir(frontSegFolder), os.listdir(rearSegFolder)):
        front_road = getRoad(front, fseg)
        rear_road = getRoad(rear, rseg)

        ftop, rtop = getTop(front_road, rear_road, front_H, rear_H)

        cv2.imsave('TOP/front/' + str(i) + '.png', ftop)
        cv2.imsave('TOP/rear/' + str(i) + '.png', rtop)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(ftop)
        ax[1].imshow(rtop)
        plt.pause(0.000001)
