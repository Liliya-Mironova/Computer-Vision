import numpy as np
import matplotlib.pyplot as plt

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
                                param1=50, param2=32, minRadius=21, maxRadius=31)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles_true = []
        
        for (x, y, r) in circles:
            if not (x < 130 or x > 3850 or y < 150 or y > 2540):
                circles_true.append((x, y, r))
                cv2.circle(gray, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(gray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
#         plot_img(gray)
#         print (len(circles_true))
        return np.array(circles_true)[:, :-1].tolist()
    
    return None

# source: https://www.geeksforgeeks.org/filter-color-with-opencv/
def find_color_mask_yellow(image, lower, upper, plot=False, color_space=cv2.COLOR_BGR2LAB):
    lab = cv2.cvtColor(image, color_space) 

    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(lab, lower, upper) 

    masked = cv2.bitwise_and(image[..., ::-1], image[..., ::-1], mask=mask) 
    if plot:
        plot_img(masked, cmap=None)
            
    return mask

# source: https://www.geeksforgeeks.org/filter-color-with-opencv/
def find_color_mask_green(image, lower, upper, plot=False, color_space=cv2.COLOR_BGR2HSV):
    hsv = cv2.cvtColor(image, color_space) 

    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(hsv, lower, upper) 

    masked = cv2.bitwise_and(image[..., ::-1], image[..., ::-1], mask=mask) 
    if plot:
        plot_img(masked, cmap=None)
            
    return mask 

# source: seminar 'contours'
def find_contours(mask, img, i_cl, i_o, div=1, plot=False):

    mask_int = mask.astype(np.uint8)
    kernel = np.ones((3, 3))
    mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel, iterations=i_cl)
    kernel = np.ones((3, 3))
    mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_OPEN, kernel, iterations=i_o)
    if plot:
        plot_img(mask_int)
    contours, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     contours = contours[1:]
    contours_true = []
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            contours_true.append(cnt)
            areas.append(area)
        
    if plot:
        img_cnt = img.copy()
        cv2.drawContours(img_cnt, contours_true, -1, (0,0,0), 10)
        plot_img(img_cnt, cmap=None)
            
    return round(sum(areas)/2800/div)

def predict_image(image):
    city_centers = find_centers(image)
    img = image[..., ::-1]
    
    mask1 = find_color_mask_green(image, [100, 140, 150], [150, 255, 255])
    mask2 = find_color_mask_green(image, [100, 135, 20], [150, 255, 255])
    mask = mask2 - mask1
    contours_blue = find_contours(mask, img, 5, 8, div=1.9)
    mask = find_color_mask_yellow(image, [73, 174, 0], [255, 255, 255])
    contours_red = find_contours(mask, img, 1, 2, div=1.47)
    mask = find_color_mask_yellow(image, [145, 0, 180], [255, 255, 255])
    contours_yellow = find_contours(mask, img, 2, 6, div=1)
    mask = find_color_mask_green(image, [55, 25, 0], [80, 255, 255])
    contours_green = find_contours(mask, img, 1, 3, div=2)
    mask = find_color_mask_green(image, [0, 0, 0], [255, 255, 20])
    contours_black = find_contours(mask, img, 6, 10, div=1.73)

#     n_trains = {'blue': 0, 'green': 0, 'black': 0, 'yellow': 0, 'red': 0}
    n_trains = {'blue': contours_blue, 'green': contours_green, 'black': contours_black, 'yellow': contours_yellow, 'red': contours_red}
    scores = {'blue': 0, 'green': 0, 'black': 0, 'yellow': 0, 'red': 0}
    return city_centers, n_trains, scores