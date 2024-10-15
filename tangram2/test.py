import cv2
import numpy as np


def find_moments(cnts, filename=None, hu_moment=True):
    # Retrieve moments of all shapes identified
    lst_moments = [cv2.moments(c) for c in cnts]
    
    # Retrieve areas of all shapes
    lst_areas = [i["m00"] for i in lst_moments]

    # Sort the contours by area (ignoring zero areas) and get the second largest
    sorted_idx = lst_areas.index(max(lst_areas))  #sorted(range(len(lst_areas)), key=lambda i: lst_areas[i], reverse=True)
    print(sorted_idx)
    # Select the second largest area
    #if len(sorted_idx) < 2:
    #    raise ValueError("Less than two valid contours available.")

    second_largest_idx = sorted_idx  # The second largest contour

    if hu_moment:  # if we want the Hu moments
        HuMo = cv2.HuMoments(lst_moments[second_largest_idx])  # grab Hu moments for the second largest shape
        if filename:
            HuMo = np.append(HuMo, filename)
        return HuMo, cnts[second_largest_idx]

    # If we want to get the moments instead
    Moms = lst_moments[second_largest_idx]
    if filename:
        Moms['target'] = filename
    return Moms, cnts[second_largest_idx]

# Functie om verschil tussen Hu moments te berekenen
def compare_hu_moments(hu1, hu2):
    # Vergelijk de Hu-momenten met de logaritmische schaal
    return np.sum(np.abs(np.log(np.abs(hu1) + 1e-10) - np.log(np.abs(hu2) + 1e-10)))

    
def detect_imgae():    
    # Stap 1: Laad de afbeelding
    image = cv2.imread('test_img.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Stap 2: Schittering maskeren (verwijder heldere gebieden)
    ret, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)

    # Stap 3: Contrast verbeteren met CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(mask)

    # Stap 4: Detecteer kleurdrempels (bijvoorbeeld rood)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = (0, 50, 50)
    upper_color = (10, 255, 255)
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Stap 5: Canny edge detection met morfologische operaties
    edges = cv2.Canny(enhanced, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Stap 6: Hooglichten en glansplekken handmatig maskeren
    ret, highlight_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    cleaned_image = cv2.bitwise_and(enhanced, enhanced, mask=cv2.bitwise_not(highlight_mask))

    # Stap 7: Vind contouren van het resultaat
    contours, _ = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100  # Example threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    moms, cnt = find_moments(filtered_contours)
    print(moms)
    for i, cnt in enumerate([cnt]):
        x,y = cnt[0,0]
        moments = cv2.moments(cnt)
        hm = cv2.HuMoments(moments)
        cv2.drawContours(image, [cnt], -1, (0,255,255), 3)
        cv2.putText(image, f'Contour {i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    # Stap 9: Toon het resultaat
    cv2.imshow('Contours', image)
    cv2.imshow('thresh', ret)

    return moms


def detect_imgae2():    
    # Stap 1: Laad de afbeelding
    image = cv2.imread('test.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Stap 2: Schittering maskeren (verwijder heldere gebieden)
    ret, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)

    # Stap 3: Contrast verbeteren met CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(mask)

    # Stap 4: Detecteer kleurdrempels (bijvoorbeeld rood)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = (0, 50, 50)
    upper_color = (10, 255, 255)
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Stap 5: Canny edge detection met morfologische operaties
    edges = cv2.Canny(enhanced, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Stap 6: Hooglichten en glansplekken handmatig maskeren
    ret, highlight_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    cleaned_image = cv2.bitwise_and(enhanced, enhanced, mask=cv2.bitwise_not(highlight_mask))

    # Stap 7: Vind contouren van het resultaat
    contours, _ = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100  # Example threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    moms, cnt = find_moments(filtered_contours)
   
    
    print(moms)
    for i, cnt in enumerate([cnt]):
        x,y = cnt[0,0]
        moments = cv2.moments(cnt)
        hm = cv2.HuMoments(moments)
        cv2.drawContours(image, [cnt], -1, (0,255,255), 3)
        cv2.putText(image, f'Contour {i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    # Stap 9: Toon het resultaat
    cv2.imshow('Contours2', image)
    cv2.imshow('thresh2', ret)

    return moms

hu1 = detect_imgae()
hu2 = detect_imgae2()
hu_difference = compare_hu_moments(hu1, hu2)
print(f"Het verschil tussen de Hu momenten van de twee tangrams is: {hu_difference}")


cv2.waitKey(0)
cv2.destroyAllWindows()
