import cv2
import numpy as np
import os
import re
import pprint
# files from the app
#from processing import preprocess_img
#from predictions import *

def delete_shadow():
    img = cv2.imread('shadows.png', -1)

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    cv2.imwrite('shadows_out.png', result)
    cv2.imwrite('shadows_out_norm.png', result_norm)

def get_countours(img):
    ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_OTSU)
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, thresh


def find_moments(cnts, filename=None, hu_moment = True):
    lst_moments = [cv2.moments(c) for c in cnts] # retrieve moments of all shapes identified
    lst_areas = [i["m00"] for i in lst_moments] # retrieve areas of all shapes
    
    try : 
        max_idx = lst_areas.index(max(lst_areas)) # select shape with the largest area

        if hu_moment: # if we want the Hu moments
            HuMo = cv2.HuMoments(lst_moments[max_idx]) # grab humoments for largest shape
            if filename:
                HuMo = np.append(HuMo, filename)
            return HuMo

        # if we want to get the moments
        Moms = lst_moments[max_idx] 
        if filename:
            Moms['target'] = filename
        return Moms
    except Exception as e:
        return [] # predictions impossible






# Write the frame to the output file
frame = cv2.imread('shadows_out.png')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cnts, thresh = get_countours(gray)
hu_moms = find_moments(cnts)

cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100]

for i, cnt in enumerate(cnts):
    x,y = cnt[0,0]
    moments = cv2.moments(cnt)
    hm = cv2.HuMoments(moments)
    cv2.drawContours(frame, [cnt], -1, (0,255,255), 3)
    cv2.putText(frame, f'Contour {i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imshow('Contours2', frame)
cv2.imshow('thresh2', thresh)


cv2.waitKey(0)

cv2.destroyAllWindows()