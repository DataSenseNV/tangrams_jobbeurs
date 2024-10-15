import cv2
import numpy as np
import os
import re
import pprint

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


# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    out.write(frame)
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cnts, thresh = get_countours(gray)
        hu_moms = find_moments(cnts)

        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100]

        for i, cnt in enumerate(cnts):
            x, y = cnt[0, 0]
            moments = cv2.moments(cnt)
            hm = cv2.HuMoments(moments)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
            cv2.putText(frame, f"Contour {i+1}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # cv2.drawContours(frame, contours, -1, (0,255,0), 20)
        cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
        cv2.imshow("Contours", frame)
        cv2.imshow("Thresh", thresh)
    except:
        print("error")

    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
