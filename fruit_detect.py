import cv2
import numpy as np
from glob import glob

def detect_fruits(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    apple = 0
    banana = 0
    orange = 0

    img_resized = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    kernel_obj_bound = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))

    left_wek = []
    right_wek = []
    top_wek = []
    bottom_wek = []

    banana_low = np.array([23, 77, 183])
    banana_top = np.array([27, 215, 234])

    orange_low = np.array([13, 157, 200])
    orange_top = np.array([18, 255, 255])

    hsv_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    thresh_object = cv2.inRange(hsv_image, (0, 114, 0), (179, 255, 255))
    closed_object = cv2.morphologyEx(thresh_object, cv2.MORPH_CLOSE, kernel=kernel_obj_bound)
    opened_object = cv2.morphologyEx(closed_object, cv2.MORPH_OPEN, kernel=kernel_obj_bound)

    contours_all, hierarchy_all = cv2.findContours(opened_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in contours_all:
        if cv2.contourArea(i) > 2000:
            x, y, w, h = cv2.boundingRect(i)
            left_wek.append(x)
            right_wek.append(x + w)
            top_wek.append(y)
            bottom_wek.append(y + h)

    thresh_banana = cv2.inRange(hsv_image, banana_low, banana_top)
    closing_banana = cv2.morphologyEx(thresh_banana, cv2.MORPH_CLOSE, kernel=kernel_obj_bound)
    opening_banana = cv2.morphologyEx(closing_banana, cv2.MORPH_OPEN, kernel=kernel_obj_bound)

    for q in range(0, len(left_wek)):
        img_temp = opening_banana[top_wek[q]:bottom_wek[q], left_wek[q]:right_wek[q]]
        contours_b, hierarchy_b = cv2.findContours(img_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_b) > 0:
            banana += 1
            left_wek[q] = -1
            right_wek[q] = -1
            top_wek[q] = -1
            bottom_wek[q] = -1

    if banana > 0:
        for app in range(0, banana):
            left_wek.remove(-1)
            right_wek.remove(-1)
            top_wek.remove(-1)
            bottom_wek.remove(-1)

    thresh_orange = cv2.inRange(hsv_image, orange_low, orange_top)
    closing_orange = cv2.morphologyEx(thresh_orange, cv2.MORPH_CLOSE, kernel=kernel_obj_bound)
    opening_orange = cv2.morphologyEx(closing_orange, cv2.MORPH_OPEN, kernel=kernel_obj_bound)

    for q in range(0, len(left_wek)):
        img_temp = opening_orange[top_wek[q]:bottom_wek[q], left_wek[q]:right_wek[q]]
        contours_o, hierarchy_o = cv2.findContours(img_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_o) > 0:
            orange += 1
            left_wek[q] = -1
            right_wek[q] = -1
            top_wek[q] = -1
            bottom_wek[q] = -1

    if orange > 0:
        for app in range(0, orange):
            left_wek.remove(-1)
            right_wek.remove(-1)
            top_wek.remove(-1)
            bottom_wek.remove(-1)

    apple = len(left_wek)


    return {'apple': apple, 'banana': banana, 'orange': orange}


def detect_all_fruits(data_path):
    img_list = glob(f'{data_path}/*.jpg')
    total_fruits = dict(total_apples=0, total_bananas=0, total_oranges=0)
    for img_path in sorted(img_list):
        fruits = detect_fruits(img_path)
        print('--------\nImage:', img_path)
        print(fruits)
        total_fruits['total_apples'] += fruits['apple']
        total_fruits['total_bananas'] += fruits['banana']
        total_fruits['total_oranges'] += fruits['orange']

    print("\n************ \n Overall:")
    print(total_fruits)

if __name__ == '__main__':
    print("##### RESULTS #####")
    detect_all_fruits('data')