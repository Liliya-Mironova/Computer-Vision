import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage import io
from skimage.feature import canny
import cv2

import os
import sys

from skimage.transform import rescale
from skimage.morphology import dilation, disk
from skimage.transform import ProjectiveTransform, warp
from skimage.feature import canny, match_template
import skimage.feature as feature

from skimage.filters import threshold_minimum, threshold_otsu, threshold_mean, threshold_local
from skimage import transform
import skimage.filters as filters
import scipy
from sudoku import *


def visualize(image_full, images_warped, preds_square, finals, templatess, corners, figsize=(20, 20)):
    image_full = image_full / 255
    
    for (image_warped, pred_square, final, templates, corner) in zip(images_warped, preds_square, finals, templatess, corners):
        solved_image = np.copy(image_warped)
        px_cell = 700 // 9
        px_cell_half = px_cell // 2

        for i in range(9):
            for j in range(9):
                if not pred_square[i, j]:
                    vcen, hcen = px_cell*i + px_cell_half, px_cell*j + px_cell_half
                    _, crop, mask, delta = templates[final[i, j]]
                    solved_image[vcen-delta:vcen+delta, hcen-delta:hcen+delta][mask] = crop[mask]

        image_warped2 = np.zeros_like(image_full)
        image_warped2[:700, :700] = np.float32(solved_image)

        mins = corner.min(axis=0)
        maxs = corner.max(axis=0)
        n = image_warped.shape[0] - 7
        points_current = ((corner - mins) / (maxs - mins) > 0.5)*n

        tform = ProjectiveTransform()
        corners_shifted = np.copy(corner)
        tform.estimate(corner, points_current)
        image_solved = warp(image_warped2, tform)

        image_full2 = np.copy(image_full)
        image_full2[image_solved > 0] = 0
        image_final = image_full2 + image_solved
        
    return image_final

from skimage.color.colorconv import rgb2gray

def draw(image, save_image_path='solution.jpg'):
    prefix = '/autograder/submission/'
#     prefix = ''
    
    sudoku_digits = []
    
    rf = joblib.load(prefix + 'random_forest.joblib')
    rf2 = joblib.load(prefix + 'random_forest2.joblib')
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    try:
        mask, corners = mask_image(image)
        images_warped, tforms = normalize_image(image)

        sudoku_digits = []
        pred_probas = []
        pred_square_masks = []
        templatess = []
        finals = []
    
        for image_warped in images_warped:
            pred_square, pred_proba, flag, cells, templates = detect_digits(image_warped, rf, get_templates=True)
            pred_square_mask = pred_square > 0

            preds_final = rf2.predict(pred_proba)

            pred_sq = np.zeros(81, dtype=np.int)
            pred_sq[flag] = preds_final
            pred_sq = pred_sq.reshape(9, 9)

            final = solve_sudoku(pred_sq)

            sudoku_digits.append(pred_sq)
            pred_probas.append(pred_proba)
            pred_square_masks.append(pred_square > 0)
            templatess.append(templates)
            finals.append(final)
  
    # visualization
        image_final = visualize(image, images_warped, pred_square_masks, finals, templatess, corners)
        cv2.imwrite(save_image_path, image_final*255)
    except:
        print (f"failed {save_image_path}")
        print("Expected error:", sys.exc_info()[0])
        pass