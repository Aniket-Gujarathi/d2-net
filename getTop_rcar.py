import cv2
import numpy as np
from PIL import Image
from Homo_rcar import Homography

img = Image.open("/scratch/udit/robotcar/2014-05-06-12-54-54/stereo/trainData_rgb/1399381468514043.png")
H = Homography(img)

img_f = Image.open('/scratch/udit/robotcar/2014-05-06-12-54-54/stereo/trainData_rgb/1399381446204705.png')
img_r = Image.open('/scratch/udit/robotcar/2014-05-06-12-54-54/rear/trainData_rgb/1399381449654770.png')

org_f = cv2.cvtColor(np.array(img_f), cv2.COLOR_BGR2RGB)
org_r = cv2.cvtColor(np.array(img_r), cv2.COLOR_BGR2RGB)

trgSize = 600
warpImg_f = cv2.warpPerspective(org_f, H, (trgSize, trgSize))
warpImg_r = cv2.warpPerspective(org_r, H, (trgSize, trgSize))

cv2.imwrite("/home/udit/d2-net/media/top_f.png", np.array(img_f))
cv2.imwrite("/home/udit/d2-net/media/top_r.png", np.array(img_r))
