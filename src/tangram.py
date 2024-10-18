import cv2
import os
import numpy as np


class TangramImage:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def get_contours(self):
        _, threshold = cv2.threshold(self.image, 1, 255, cv2.THRESH_OTSU)
        self.contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return self.contours

    @staticmethod
    def find_moments(contours):
        moments = [cv2.moments(c) for c in contours]
        areas = [i["m00"] for i in moments]

        index_of_largest_area = areas.index(max(areas))

        target_moment = moments[index_of_largest_area]
        target_contour = contours[index_of_largest_area]

        return target_moment, target_contour

    @staticmethod
    def find_hu_moments(contours):
        moments = [cv2.moments(c) for c in contours]
        areas = [i["m00"] for i in moments]

        index_of_largest_area = areas.index(max(areas))

        target_moment = moments[index_of_largest_area]
        target_moment = cv2.HuMoments(target_moment)
        target_contour = contours[index_of_largest_area]

        return target_moment, target_contour

    @staticmethod
    def compare_hu_moments(hu_input, hu_target):
        return np.sum(
            np.abs(np.log(np.abs(hu_input) + 1e-10) - np.log(np.abs(hu_target) + 1e-10))
        )

    def process(self, min_area=100):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(mask)
        self.edges = cv2.Canny(enhanced, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.morphed_edges = cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, kernel)
        self.contours, _ = cv2.findContours(
            self.morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered_contours = [
            cnt for cnt in self.contours if cv2.contourArea(cnt) > min_area
        ]
        self.hu_moments, _ = TangramImage.find_hu_moments(filtered_contours)
        return self.hu_moments

    def draw_contours(self, path=None):
        contoured_image = self.image.copy()
        for i, cnt in enumerate(self.contours):
            cv2.drawContours(contoured_image, [cnt], -1, (0, 255, 255), 3)
            cv2.putText(
                contoured_image,
                f"Contour {i+1}",
                (cnt[0, 0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        self.contoured_image = contoured_image
        if path:
            cv2.imwrite(path, self.contoured_image)

    def show_contours(self):
        if not hasattr(self, "contoured_image"):
            raise Exception("No contours drawn yet.")
        cv2.imshow(
            "Contours {}".format(self.image_path.split(os.sep).pop()),
            self.contoured_image,
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
