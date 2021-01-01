import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_holes, remove_small_objects
from sklearn.cluster import OPTICS

import cv2

# source: seminar 'lamp_detection'
def plot_img(img, cmap='gray'):
    plt.figure(figsize=(14, 8))
    plt.imshow(img, cmap=cmap)
#     plt.axis('off')
    plt.show()

# source: https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
def find_centers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                param1=50, param2=32, minRadius=22, maxRadius=31)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles_true = []
        
        for (x, y, r) in circles:
            if not (x < 130 or x > 3850 or y < 150 or y > 2540):
                circles_true.append((x, y, r))
                cv2.circle(gray, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(gray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        plot_img(gray)
        print (len(circles_true))
        return np.array(circles_true)[:, :-1].tolist()
    
    return None

# source: https://www.geeksforgeeks.org/filter-color-with-opencv/
def find_color_mask(image, lower, upper, v1, v2, plot=False, color_space=cv2.COLOR_BGR2HSV):
    hsv = cv2.cvtColor(image, color_space) 

    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(hsv, lower, upper)
    arr = mask > 0
    mask = remove_small_objects(arr, min_size=v1).astype(np.uint8)
    mask = remove_small_holes(mask, min_size=v2).astype(np.uint8)

    masked = cv2.bitwise_and(image[..., ::-1], image[..., ::-1], mask=mask) 
    if plot:
        plot_img(mask, cmap=None)
    
    return masked

# source: seminar 'contours'
def find_contours(image, img, v1, v2, plot=False):
    HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]
    mask = (HUE < 10) | (HUE > 180)

    mask_int = mask.astype(np.uint8)
    kernel = np.ones((v1, v1))
    mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((v2, v2))
    mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel)
    plot_img(mask_int)

    contours, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = contours[1:]
#     contours_true = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 500:
#             contours_true.append(cnt)
            
    if plot:
        centers = []
        for c in contours:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
            
        clustering = OPTICS(min_samples=2).fit(np.array(centers))
        cls = clustering.labels_
        
        img_cnt = img.copy()
        cv2.drawContours(img_cnt, contours, -1, (0,0,0), 10)
        for c, cl, cll in zip(contours, centers, cls):
            cv2.circle(img_cnt, (cl[0], cl[1]), 7, (255, 255, 255), -1)
            cv2.putText(img_cnt, str(cll), (cl[0] - 20, cl[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
        
        plot_img(img_cnt, cmap=None)
       
            
    return len(contours)