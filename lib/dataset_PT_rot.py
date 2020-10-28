import numpy as np
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sys import exit, argv
import csv
import torch
#import cv2
from scipy import ndimage
from torch.utils.data import Dataset
from lib.utils import preprocess_image


class PhotoTourism(Dataset):
    def __init__(self, images):
        self.images = images
        self.dataset = []

    def getImageFiles(self):
        imgFiles = []

        with open(self.images) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for i, row in enumerate(csvReader):
                imgFiles.append(row)
				#print(imgFiles)
        return imgFiles

    def img_rotn(self, img1):
        np.random.seed(0)
        img2 = img1.rotate( np.random.randint(low = 0, high = 360))
        img2.save("img2.jpg")

        return img2

    def build_dataset(self):
        self.dataset = []
        print('Building dataset')

        imgFiles = self.getImageFiles()
        for img in (imgFiles):
            #print(img[1])
            img1 = Image.open(img[1])
            img2 = PhotoTourism.img_rotn(self, img1)
            #img2 = Image.open(img2)

            if(img1.mode != 'RGB'):
                img1 = img1.convert('RGB')

            if(img2.mode != 'RGB'):
                img2 = img2.convert('RGB')

            img1 = np.array(img1)
            img2 = np.array(img2)

            self.dataset.append((img1, img2))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        image1, image2 = self.dataset[idx]

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)
        print('hi', len(self.dataset))

        return {
			'image1': torch.from_numpy(image1.astype(np.float32)),
			'image2': torch.from_numpy(image2.astype(np.float32)),
			}

if __name__ == '__main__':
	# rootDir = "/scratch/udit/"
    #rootDir = argv[1]
	images = argv[1]

	training_dataset = PhotoTourism(images)

	training_dataset.build_dataset()

