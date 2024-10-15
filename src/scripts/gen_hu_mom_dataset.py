import cv2
import numpy as np
import os
import re
import pprint
import pickle

# files from the app
# from processing import preprocess_img
# from predictions import *


def get_countours(img):
    ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_OTSU)
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, thresh


def find_moments(cnts, filename=None, hu_moment=True):
    lst_moments = [cv2.moments(c) for c in cnts]  # retrieve moments of all shapes identified
    lst_areas = [i["m00"] for i in lst_moments]  # retrieve areas of all shapes

    try:
        max_idx = lst_areas.index(max(lst_areas))  # select shape with the largest area

        if hu_moment:  # if we want the Hu moments
            HuMo = cv2.HuMoments(lst_moments[max_idx])  # grab humoments for largest shape
            if filename:
                HuMo = np.append(HuMo, filename)
            return HuMo

        # if we want to get the moments
        Moms = lst_moments[max_idx]
        if filename:
            Moms["target"] = filename
        return Moms
    except Exception as e:
        return []  # predictions impossible


def find_moments(cnts, filename=None, hu_moment=True):
    # Retrieve moments of all shapes identified
    lst_moments = [cv2.moments(c) for c in cnts]

    # Retrieve areas of all shapes
    lst_areas = [i["m00"] for i in lst_moments]

    try:
        # Sort the contours by area (ignoring zero areas) and get the second largest
        sorted_idx = sorted(range(len(lst_areas)), key=lambda i: lst_areas[i], reverse=True)

        # Select the second largest area
        if len(sorted_idx) < 2:
            raise ValueError("Less than two valid contours available.")

        second_largest_idx = sorted_idx[1]  # The second largest contour

        if hu_moment:  # if we want the Hu moments
            HuMo = cv2.HuMoments(lst_moments[second_largest_idx])  # grab Hu moments for the second largest shape
            if filename:
                HuMo = np.append(HuMo, filename)
            return HuMo, cnts[second_largest_idx]

        # If we want to get the moments instead
        Moms = lst_moments[second_largest_idx]
        if filename:
            Moms["target"] = filename
        return Moms, cnts[second_largest_idx]

    except Exception as e:
        return []  #


img = cv2.imread("./dataset/cat.png")

# Write the frame to the output file
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cnts, thresh = get_countours(gray)
hu_moms, cnt = find_moments(cnts)

cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100]
# moments, cnt = cv2.moments(cnts)

mom = find_moments(cnts)

cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)
x, y = cnt[0, 0]

cv2.putText(img, f"Contour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imwrite("test.png", img)

print(hu_moms)
# pickle.dump(hu_moms)
