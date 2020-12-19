import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage import io
from skimage.feature import canny
import cv2
import os

import numpy as np
import cv2
import os

from skimage.feature import hog

from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

from skimage.transform import rescale
from skimage.morphology import dilation, disk
from skimage.transform import ProjectiveTransform, warp
from skimage.feature import canny, match_template
import skimage.feature as feature

from skimage.filters import threshold_minimum, threshold_otsu, threshold_mean, threshold_local
from skimage import transform
import skimage.filters as filters
import scipy
import joblib

#---------------------------------------------------------------
# SUDOKU SOLVER FROM GITHUB

def matrix_to_puzzle(matrix):
    rows = [' {}  {}  {} | {}  {}  {} | {}  {}  {}'.format(*x).replace('0', '.') for x in matrix]
    rows.insert(6, '---------+---------+---------')
    rows.insert(3, '---------+---------+---------')
    return '\n'.join(rows)

def puzzle_to_matirx(puzzle):
    vector = [x for x in puzzle.values()]
    return np.array(vector).astype(int).reshape([9, 9])

def solve_sudoku(sudoku_matrix):
    puzzle = matrix_to_puzzle(sudoku_matrix)
    try:
        solution = solve_puzzle(puzzle)
    except TypeError as e:
        raise ValueError('An error in sudoku solving engine. Please check your CV algorithm!') from None
    return puzzle_to_matirx(solution)

################### The code above converts matrix to string and back

################### The code below is taken from https://medium.com/@neshpatel/solving-sudoku-part-i-7c4bb3097aa7


def sudoku_reference():
    """Gets a tuple of reference objects that are useful for describing the Sudoku grid."""
    
    def cross(str_a, str_b):
        """Cross product (concatenation) of two strings A and B."""
        return [a + b for a in str_a for b in str_b]

    all_rows = 'ABCDEFGHI'
    all_cols = '123456789'

    # Build up list of all cell positions on the grid
    coords = cross(all_rows, all_cols)
    # print(len(coords))  # 81

    # Get the units for each row
    row_units = [cross(row, all_cols) for row in all_rows]
    # print(row_units[0])  # ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']

    # Do it in reverse to get the units for each column
    col_units = [cross(all_rows, col) for col in all_cols]
    # print(col_units[0])  # ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1']

    box_units = [cross(row_square, col_square) for row_square in ['ABC', 'DEF', 'GHI'] for col_square in ['123', '456', '789']]
    # print(box_units[0])   # ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']

    all_units = row_units + col_units + box_units  # Add units together
    groups = {}

    # For each cell, get the each unit that the cell is part of (3 per cell)
    groups['units'] = {pos: [unit for unit in all_units if pos in unit] for pos in coords}
    # print(units['A1'])  # 3 Units of length 9 for each cell

    # For each cell get the list of peers to that position
    groups['peers'] = {pos: set(sum(groups['units'][pos], [])) - {pos} for pos in coords}
    # print(peers['A1'])  # Peer cells for the position, length 20 for each cell
    
    return coords, groups, all_units


def parse_puzzle(puzzle, digits='123456789', nulls='0.'):
    """
    Parses a string describing a Sudoku puzzle board into a dictionary with each cell mapped to its relevant
    coordinate, i.e. A1, A2, A3...
    """

    # Serialise the input into a string, let the position define the grid location and .0 can be empty positions
    # Ignore any characters that aren't digit input or nulls
    flat_puzzle = ['.' if char in nulls else char for char in puzzle if char in digits + nulls]

    if len(flat_puzzle) != 81:
        raise ValueError('Input puzzle has %s grid positions specified, must be 81. Specify a position using any '
                         'digit from 1-9 and 0 or . for empty positions.' % len(flat_puzzle))

    coords, groups, all_units = sudoku_reference()

    # Turn the list into a dictionary using the coordinates as the keys
    return dict(zip(coords, flat_puzzle))

def validate_sudoku(puzzle):
    """Checks if a completed Sudoku puzzle has a valid solution."""
    
    if puzzle is None:
        return False
      
    coords, groups, all_units = sudoku_reference()
    full = [str(x) for x in range(1, 10)]  # Full set, 1-9 as strings
    
    # Checks if all units contain a full set
    return all([sorted([puzzle[cell] for cell in unit]) == full for unit in all_units])


def solve_puzzle(puzzle):
    """Solves a Sudoku puzzle from a string input."""
    digits = '123456789'  # Using a string here instead of a list

    coords, groups, all_units = sudoku_reference()
    input_grid = parse_puzzle(puzzle)
    input_grid = {k: v for k, v in input_grid.items() if v != '.'}  # Filter so we only have confirmed cells
    output_grid = {cell: digits for cell in coords}  # Create a board where all digits are possible in each cell

    def confirm_value(grid, pos, val):
        """Confirms a value by eliminating all other remaining possibilities."""
        remaining_values = grid[pos].replace(val, '')  # Possibilities we can eliminate due to the confirmation
        for val in remaining_values:
            grid = eliminate(grid, pos, val)
        return grid

    def eliminate(grid, pos, val):
        """Eliminates `val` as a possibility from all peers of `pos`."""

        if grid is None:  # Exit if grid has already found a contradiction
            return None

        if val not in grid[pos]:  # If we have already eliminated this value we can exit
            return grid

        grid[pos] = grid[pos].replace(val, '')  # Remove the possibility from the given cell

        if len(grid[pos]) == 0:  # If there are no remaining possibilities, we have made the wrong decision
            return None
        elif len(grid[pos]) == 1:  # We have confirmed the digit and so can remove that value from all peers now
            for peer in groups['peers'][pos]:
                grid = eliminate(grid, peer, grid[pos])  # Recurses, propagating the constraint
                if grid is None:  # Exit if grid has already found a contradiction
                    return None

        # Check for the number of remaining places the eliminated digit could possibly occupy
        for unit in groups['units'][pos]:
            possibilities = [p for p in unit if val in grid[p]]

            if len(possibilities) == 0:  # If there are no possible locations for the digit, we have made a mistake
                return None
            # If there is only one possible position and that still has multiple possibilities, confirm the digit
            elif len(possibilities) == 1 and len(grid[possibilities[0]]) > 1:
                if confirm_value(grid, possibilities[0], val) is None:
                    return None

        return grid

    # First pass of constraint propagation
    for position, value in input_grid.items():  # For each value we're given, confirm the value
        output_grid = confirm_value(output_grid, position, value)

    if validate_sudoku(output_grid):  # If successful, we can finish here
        return output_grid

    def guess_digit(grid):
        """Guesses a digit from the cell with the fewest unconfirmed possibilities and propagates the constraints."""

        if grid is None:  # Exit if grid already compromised
            return None

        # Reached a valid solution, can end
        if all([len(possibilities) == 1 for cell, possibilities in grid.items()]):
            return grid

        # Gets the coordinate and number of possibilities for the cell with the fewest remaining possibilities
        n, pos = min([(len(possibilities), cell) for cell, possibilities in grid.items() if len(possibilities) > 1])

        for val in grid[pos]:
            # Run the constraint propagation, but copy the grid as we will try many adn throw the bad ones away.
            # Recursively guess digits until its complete and there's a valid solution
            solution = guess_digit(confirm_value(grid.copy(), pos, val))
            if solution is not None:
                return solution

    output_grid = guess_digit(output_grid)
    return output_grid

# -------------------------------------------------------------------
# MY CODE
from skimage.morphology import dilation, disk

def mask_image(image_full, verbose=False, return_corners=False):
    image_scaled = np.copy(image_full)
    image_scaled = 255 - image_scaled
    
    edges = image_scaled
    edges = canny(edges, sigma=2)
    edges = np.int8(filters.gaussian(edges, 2) * 255)
    selem = disk(1)
    edges = dilation(edges, selem)

    # we need to convert image to uint to apply findContours
    edges = (edges).astype(np.uint8)
    ext_contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    

    areas = [cv2.contourArea(x) for x in ext_contours]
    contours = []
    for i, a in enumerate(areas):
        if a > 0.7*max(areas):
            contours.append(ext_contours[i])

    # we need to remove one unnecessary dimension
    contours = [c.squeeze() for c in contours]

    contours_real = []
    for cont in contours:
        maxs = cont.max(axis=0)
        mins = cont.min(axis=0)
        a = maxs[0] - mins[0]
        b = maxs[1] - mins[1]
        min_, max_ = min(a, b), max(a, b)
        arc_length_area = cv2.arcLength(cont, True)**2/cv2.contourArea(cont)/16
    
        if min_ / max_ > 0.8 and arc_length_area < 1.5:
            contours_real.append(cont)
        
    # epsilon allow us to control the max deviation from the original curve 
    # here we use 5% of the total curve lenth 
    corners = []
    for cnt in contours_real:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        corners += [cv2.approxPolyDP(cnt, epsilon, True).squeeze()]
       
    mask = np.zeros_like(image_scaled)
    cv2.fillPoly(mask, corners, 1)
    
    return mask.astype(np.bool_), corners

def normalize_image(image_full, verbose=False, return_tform=False):
    image_scaled = rescale(image_full, 0.3, multichannel=False)
    edges = canny(image_scaled,
                  sigma=1.1,
                  low_threshold=0.66*np.mean(image_scaled),
                  high_threshold=1.33*np.mean(image_scaled))
    edges = dilation(edges, disk(1))
    
    edges = (edges).astype(np.uint8)
    ext_contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    areas = [cv2.contourArea(x) for x in ext_contours]
    contours = []
    for i, a in enumerate(areas):
        if a > 0.7*max(areas):
            contours.append(ext_contours[i])

    # we need to remove one unnecessary dimension
    contours = [c.squeeze() for c in contours]

    contours_real = []
    for cont in contours:
        maxs = cont.max(axis=0)
        mins = cont.min(axis=0)
        a = maxs[0] - mins[0]
        b = maxs[1] - mins[1]
        min_, max_ = min(a, b), max(a, b)
        if min_ / max_ > 0.8:
            contours_real.append(cont)
        
    # epsilon allow us to control the max deviation from the original curve 
    # here we use 5% of the total curve lenth 
    
    images_warped = []
    tforms = []
    for cnt in contours_real:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        corners = cv2.approxPolyDP(cnt, epsilon, True).squeeze()

        desired = [0, 0, 0, 0]
        desired[np.argmin(corners.sum(axis=1))] = [0, 0]
        desired[np.argmax(corners.sum(axis=1))] = [693, 693]
        idx_left = {0, 1, 2, 3} - {np.argmin(corners.sum(axis=1)), np.argmax(corners.sum(axis=1))}
        topl = list(idx_left)[np.argmin(corners[list(idx_left)][:, 0])]
        topr = list(idx_left)[np.argmax(corners[list(idx_left)][:, 0])]
        desired[topl] = [0, 693]
        desired[topr] = [693, 0]

        points_current = corners
        points_desired = np.array(desired)

        tform = ProjectiveTransform()
        tform.estimate(points_desired, points_current)
        image_warped = warp(image_scaled, tform)[:700, :700]
        
        images_warped.append(image_warped)
        tforms.append(tforms)
    
    return images_warped, tforms

def detect_digits(image_warped, rf, get_templates=False):
    shift = -10
    px_cell = 700 // 9
    px_cell_part = px_cell // 3
    full_size = 700

    process_cell(image_warped, (0, 2), px_cell, shift, px_cell // 4, plotit=False);

    cells = np.zeros((9, 9, 28, 28))
    delta = np.zeros((9, 9))

    for i in range(9):
        for j in range(9):
            hmin, hmax = px_cell*j+px_cell_part, px_cell*(j+1)-px_cell_part
            vmin, vmax = px_cell*i+px_cell_part, px_cell*(i+1)-px_cell_part
            crop = 1 - np.copy(image_warped[vmin:vmax, hmin:hmax])
            delta[i, j] = crop.max() - crop.min()

    delta_threshold = threshold_minimum(delta)
    flag = delta > delta_threshold

    for i in range(9):
        for j in range(9):
            if flag[i, j]:
                crop = process_cell(image_warped, (i, j), px_cell, shift, px_cell_part)
            else:
                crop = np.zeros((28, 28))
            cells[i, j] = crop

    cells = cells.reshape(-1, *(cells.shape[-2:]))
    flag = cells.mean(axis=(-1, -2)) > 0
    hhog = lambda image: hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), visualize=False, multichannel=False)
    features_cell = np.array([hhog(im) for im in cells[flag]])

    pred_proba = rf.predict_proba(features_cell)
    pred = np.argmax(pred_proba, axis=1)

    pred_square = np.zeros(81, dtype=np.int)
    pred_square[flag] = pred
    pred_square = pred_square.reshape(9, 9)
    
    templates = {}
    if get_templates:
        for k in range(1, 10):
            i, j = np.vstack(np.where(pred_square == k)).T[0]
            vmin, vmax, hmin, hmax = np.clip([px_cell*i+shift, px_cell*(i+1)-shift, px_cell*j+shift, px_cell*(j+1)-shift], 0, 700)
            crop_small, mask, (v_center, h_center, delta) = process_cell(image_warped, (i, j), px_cell, shift, px_cell_part, return_crop_position=True)
            v_center += vmin
            h_center += hmin
            crop = np.copy(image_warped[v_center-delta:v_center+delta, h_center-delta:h_center+delta])
            templates[k] = (crop_small, crop, mask, delta)

    return pred_square, pred_proba, flag, cells, templates

def process_cell(image_warped, ij, px_cell, shift, px_cell_part, plotit=False, return_crop_position=False):
    i, j = ij
    full_size = 700
    hmin, hmax = px_cell*j+shift, px_cell*(j+1)-shift
    vmin, vmax = px_cell*i+shift, px_cell*(i+1)-shift
    v_off_beg, v_off_end, h_off_beg, h_off_end = 0, 0, 0, 0

    if hmin < 0:
        h_off_beg = -hmin
        hmin = 0
    if hmax > full_size:
        h_off_end = full_size - hmax
        hmax = full_size
    if vmin < 0:
        v_off_beg = -vmin
        vmin = 0
    if vmax > full_size:
        v_off_end = full_size - vmax
        vmax = full_size
        vmin -= v_off_end

    crop = np.copy(image_warped[vmin:vmax, hmin:hmax])
    crop = 1. - crop
    if plotit:
        plt.imshow(crop)
        plt.colorbar()
        plt.title("Initial")
        plt.show()

        crop = filters.gaussian(crop, 1)
    if plotit:
        plt.imshow(crop)
        plt.colorbar()
        plt.title("With gaussian bluering")
        plt.show()

    offset = px_cell_part-shift
    crop_small = np.copy(crop[offset-v_off_beg:-(offset-v_off_end), offset-h_off_beg:-(offset-h_off_end)])
    crop_threshold = threshold_minimum(crop_small)
    if plotit:
        plt.imshow(crop_small)
        plt.colorbar()
        plt.title("Center part")
        plt.show()
        plt.imshow(1* (crop > crop_threshold))
        plt.colorbar()
        plt.title("Binarization")
        plt.show()

    labels, n = scipy.ndimage.label(crop > crop_threshold)
    if plotit:
        plt.imshow(labels)
        plt.colorbar()
        plt.title("Segmentation")
        plt.show()
        
    correct_labels = np.unique(labels[offset-v_off_beg:-(offset-v_off_end), offset-h_off_beg:-(offset-h_off_end)])
    correct_labels = correct_labels[correct_labels > 0]
    correct_mask = np.isin(labels, correct_labels)
    background_level = np.mean(crop[labels == 0])
    crop[~correct_mask] = background_level
    if plotit:
        plt.imshow(crop)
        plt.colorbar()
        plt.title("Remove other labels")
        plt.show()


    vrange = np.nonzero(correct_mask.any(axis=1))[0]
    v_center = (vrange[0] + vrange[-1]) // 2
    v_delta = vrange[-1] - vrange[0]
    hrange = np.nonzero(correct_mask.any(axis=0))[0]
    h_center = (hrange[0] + hrange[-1]) // 2
    h_delta = hrange[-1] - hrange[0]
    delta = np.max((v_delta, h_delta)) // 2 + 5
    crop = crop[v_center-delta:v_center+delta, h_center-delta:h_center+delta]
    if return_crop_position:
        crop_threshold = threshold_minimum(crop)
        mask = crop > crop_threshold
    if plotit:
        plt.imshow(crop)
        plt.colorbar()
        plt.title("Center crop")
        plt.show()

    crop = transform.resize(crop, (28, 28))
    if plotit:
        plt.imshow(crop)
        plt.colorbar()
        plt.title("Resize")
        plt.show()

    crop_min, crop_max = crop.min(), crop.max()
    crop = (crop - crop_min) / (crop_max - crop_min)
   
    if plotit:
        plt.imshow(crop)
        plt.colorbar()
        plt.title("Rescale")
        plt.show()

    if return_crop_position:
        return crop, mask, (v_center, h_center, delta)
    return crop

def predict_image(image):
    prefix = '/autograder/submission/'
#     prefix = ''

    sudoku_digits = []
    
    rf = joblib.load(prefix + 'random_forest.joblib')
    rf2 = joblib.load(prefix + 'random_forest2.joblib')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask, corners = mask_image(image)
    images_warped, tforms = normalize_image(image)
    
    sudoku_digits = []
    for image_warped in images_warped:
        pred_square, pred_proba, flag, cells, templates = detect_digits(image_warped, rf, get_templates=False)
        pred_square_mask = pred_square > 0
        
        preds_final = rf2.predict(pred_proba)

        pred_sq = np.zeros(81, dtype=np.int)
        pred_sq[flag] = preds_final
        pred_sq = pred_sq.reshape(9, 9)
        pred_sq[pred_sq == 0] = -1

        sudoku_digits.append(pred_sq)
    
    return mask, sudoku_digits