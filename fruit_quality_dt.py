import cv2
import numpy as np

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

img = cv2.imread('data/rotten_apple_1.jpg')
cv2.imshow('image', img)

# converting BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

# create a red HSV colour boundary and
# threshold HSV image
redmask1 = cv2.inRange(hsv, lower_red, upper_red)

# define range of red color in HSV
lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])

# create a red HSV colour boundary and
# threshold HSV image
redmask2 = cv2.inRange(hsv, lower_red, upper_red)

redmask = redmask1  # + redmask2

maskOpen = cv2.morphologyEx(redmask, cv2.MORPH_OPEN, kernelOpen)
maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

maskFinal = maskClose
cv2.imshow('Red_Mask:', maskFinal)

cnt_r = 0
for r in redmask:
    cnt_r = cnt_r + list(r).count(255)
print("Redness ", cnt_r)

lower_green = np.array([50, 50, 50])
upper_green = np.array([70, 255, 255])

greenmask = cv2.inRange(hsv, lower_green, upper_green)
cv2.imshow('Green_Mask:', greenmask)
cnt_g = 0
for g in greenmask:
    cnt_g = cnt_g + list(g).count(255)
print("Greenness ", cnt_g)

# lower_yellow=np.array([20,50,50])
# upper_yellow=np.array([30,255,255])

# yellowmask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# cv2.imshow('Yellow_Mask:',yellowmask)
# cnt_y=0
# for y in yellowmask:
#    cnt_y=cnt_y+list(y).count(255)
# print ("Yellowness ",cnt_y)

lower_brown = np.array([170, 50, 50])
upper_brown = np.array([180, 255, 255])

rotten_mask = cv2.inRange(hsv, lower_brown, upper_brown)
cv2.imshow('Yellow_Mask:', rotten_mask)
cnt_y = 0
for y in rotten_mask:
    cnt_y = cnt_y + list(y).count(255)
print("Yellowness ", cnt_y)

# Calculate ripeness
tot_area = cnt_r + cnt_y + cnt_g

red_perc = cnt_r / tot_area
rotten_perc = cnt_y / tot_area
green_perc = cnt_g / tot_area

# Adjust the limits for your fruit
green_limit = 0.5
red_limit = 0.8

if green_perc > green_limit:
    print("Unripess")
elif red_perc > red_limit:
    print("Ripeness")
else:
    print("Rotten")

# De-allocate any associated memory usage
cv2.waitKey(0)
cv2.destroyAllWindows()
