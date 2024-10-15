import cv2
import numpy as np

img = cv2.imread("tangram_creations/cat_1.jpg", -1)


def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


# Shadow removal
shad = shadow_remove(img)

cv2.imwrite("after_shadow_remove13.jpg", shad)

img = cv2.imread("after_shadow_remove13.jpg", -1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))
nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))
nzmask = cv2.erode(nzmask, np.ones((3, 3)))
mask = mask & nzmask
new_img = img.copy()
new_img[np.where(mask)] = 255


cv2.imshow("mask", mask)
cv2.imshow("new_img", new_img)
cv2.imwrite("after_shadow_remove13.jpg", new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
