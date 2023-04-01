import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#remind me to never do this again thanks

path ='data/red_apple.jpg'
img = cv.imread(path)

img = cv.medianBlur(img,7)
cv.imshow('blurred image',img)

Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
print(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imshow('KMeans Clustering',res2)

gri = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
(thresh1, I1) = cv.threshold(gri,127,255,cv.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(I1,kernel,iterations = 0)
dilation = cv.dilate(I1,kernel,iterations = 1)

plt.subplot(131), plt.imshow(img, 'gray'),plt.title('Imagine originala')
plt.subplot(132), plt.imshow(erosion,'gray'),plt.title('Imagine erodata')
plt.subplot(133), plt.imshow(dilation,'gray'),plt.title('Imagine dilatata')
plt.show()
cv.waitKey(1000)

number_of_white_pix = np.sum(erosion == 255)
number_of_black_pix = np.sum(erosion == 0)
total_pix= number_of_white_pix + number_of_black_pix

print('Total pixel:', total_pix)
print('Number of white pixels:', number_of_white_pix)
print('Number of black pixels:', number_of_black_pix)

white_percent= number_of_white_pix / total_pix * 100
black_percent= number_of_black_pix / total_pix * 100

print('Total ', white_percent, '% white pixel')
print('Sunt ', black_percent, '% black pixel')


if(black_percent>5 and black_percent!=0):
    img = cv.putText(img, 'Rotten apple!', (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                     0.7, (255, 0, 0), 1, cv.LINE_AA)
else:
    img = cv.putText(img, 'Perfect apple!', (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                     0.7, (255, 0, 0), 1, cv.LINE_AA)

cv.imshow('Rezultat', img)

cv.waitKey(0)