import cv2
  
image = cv2.imread('1403774746039904.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original image',image)
#cv2.imshow('Gray image', gray)
cv2.imwrite('gray2.png', gray)

cv2.destroyAllWindows()
