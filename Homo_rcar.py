from PIL import Image
import cv2
import numpy as np

def Homography(img):

    rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    # cv2.imshow("Im", rgb)
    # cv2.waitKey(0)
    
    # bottom left -> bottom right -> top right -> top left
    srcPts = np.array([[460, 774], [1119, 774], [919, 628], [534, 620]])
    trgPts = np.array([[0, 600], [600, 600], [600, 0], [0, 0]])

    for i in range(0, len(srcPts)):
        rgb = cv2.circle(rgb, (srcPts[i][0], srcPts[i][1]), 1, (0, 0, 255), 2)
    # cv2.imshow("Im", rgb)
    # cv2.waitKey(0)

    for i in range(0, len(trgPts)):
        rgb = cv2.circle(rgb, (trgPts[i][0], trgPts[i][1]), 1, (255, 0, 0), 2)
    # cv2.imshow("Im", rgb)
    # cv2.waitKey(0)

    homographyMat, status = cv2.findHomography(srcPts, trgPts)

    return homographyMat

if __name__ == '__main__':
    img = Image.open("/scratch/udit/robotcar/2014-05-06-12-54-54/stereo/trainData_rgb/1399381468514043.png")
    H = Homography(img)
    trgSize = 600
    orgImg = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    warpImg = cv2.warpPerspective(orgImg, H, (trgSize, trgSize))

    cv2.imshow("Warped", warpImg)
    cv2.waitKey(0)
