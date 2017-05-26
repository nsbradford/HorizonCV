""" 
    horizon.py
    Nicholas S. Bradford

"""

import cv2
import numpy as np
import math

DEBUG_VERBOSE = False
def printV(text):
    if DEBUG_VERBOSE:
        print(text)

def convert_m_b_to_pitch_bank(m, b, sigma_below):
    """ Method from original paper:
            Pitch angle (Theta) = size(ground) / size(ground) + size(sky)
            Bank angle (Phi) = tan^-1(m)
        Args:
            m
            b
            sigma_below
        Returns:
            pitch
            bank
    """
    bank = math.degrees(math.atan(m))
    pitch = sigma_below
    return pitch, bank


# def convert_pitch_roll_to_m_b(pitch, bank):
#     """ 
#         Limits of (pitch, roll) space are [-pi/2, pi/2] for pitch and [0%, 100%] for roll
#     """
#     m = math.tan(bank) #TODO
#     b = None
#     return (m, b)


# TODO more efficient implementation
# def img_line_mask2(rows, columns, m, b):
#     """ Params:
#             rows (int)
#             columns (int)
#             m (double)
#             b (double)
#         Returns:
#             rows x columns np.array boolean mask with True for all values above the line
#     """
#     mask = np.zeros((rows, columns), dtype=np.bool)
#     for x in range(columns):
#         y = m * x + b
#         # ind = np.arange(0, min(int(y), int(rows)))
#         ind = np.arange(max(0, int(y)), int(rows))
#         # print(y, ind)
#         mask[ind, x] = True
#     return mask


def img_line_mask(rows, columns, m, b):
    """ Params:
            rows (int)
            columns (int)
            m (double)
            b (double)
        Returns:
            rows x columns np.array boolean mask with True for all values above the line
    """
    mask = np.zeros((rows, columns), dtype=np.bool)
    for y in range(rows):
        for x in range(columns):
            if y >= m * x + b:
                mask[y, x] = True
    return mask


def split_img_by_line(img, m, b):
    """ Params:
            m: slope
            b: y-intercept
        Returns:
            (arr1, arr2): two np.arrays with 3 columns (RGB) and N rows (one for each pixel)
    """
    mask = img_line_mask(rows=img.shape[0], columns=img.shape[1], m=m, b=b)
    assert len(mask.shape) == 2
    assert mask.shape[0] == img.shape[0]
    assert mask.shape[1] == mask.shape[1]
    segment1 = img[mask]
    segment2 = img[np.logical_not(mask)]
    reshape1 = segment1.reshape(-1, segment1.shape[-1])
    reshape2 = segment2.reshape(-1, segment2.shape[-1])
    assert reshape1.shape[1] == reshape2.shape[1] == 3
    if not (segment1.shape[0] > 1 and segment2.shape[0] > 1): 
        print('!!    Warning: Invalid hypothesis: ' + str(m) + ' ' + str(b))
    return (reshape1, reshape2)


def compute_variance_score(segment1, segment2):
    """ Params:
            segment1 (np.array): n x 3, where n is number of pixels in first segment
            segment2 (np.array): n x 3, where n is number of pixels in first segment
        Returns:
            F (np.double): the score for these two segments (higher = better line hypothesis)
        Note that linalg.eigh() is more stable than np.linalg.eig, but only for symmetric matrices.
    """
    assert segment1.shape[1] == segment2.shape[1] == 3
    if not (segment1.shape[0] > 1 and segment2.shape[0] > 1):
        return -1.0
    cov1 = np.cov(segment1.T)
    cov2 = np.cov(segment2.T)
    assert cov1.shape == cov2.shape == (3,3)
    evals1, evecs1 = np.linalg.eigh(cov1)
    evals2, evecs2 = np.linalg.eigh(cov2)
    det1 = evals1.prod() #np.linalg.det(cov1)
    det2 = evals2.prod() #np.linalg.det(cov2)
    score =  det1 + det2 + (np.sum(evals1) ** 2) + (np.sum(evals2) ** 2)
    return score ** -1


def score_line(img, m, b):
    """
        Params:
            img
            m
            b
    """
    # print('Score', img.shape, m, b)
    seg1, seg2 = split_img_by_line(img, m=m, b=b)
    # print('\tSegment shapes: ', seg1.shape, seg2.shape)
    score = compute_variance_score(seg1, seg2)
    return score


def score_grid(img, grid):
    scores = list(map(lambda x: score_line(img, x[0], x[1]), grid))
    assert len(scores) > 0, 'Invalid slope and intercept ranges: '
    max_index = np.argmax(scores)
    m, b = grid[max_index]
    return m, b, scores, grid, scores[max_index]


def accelerated_search(img, m, b, current_score):
    max_iter = 10
    delta_m = 0.25
    delta_b = 1.0
    delta_factor = 0.75
    printV('\tAccel. search begin m: {} b: {}'.format(m, b))
    for i in range(max_iter):
        # print('\tDelta', delta_m, delta_b, 'M&B', m, b)
        grid = [
                    (m + delta_m, b),
                    (m - delta_m, b),
                    (m, b + delta_b),
                    (m, b - delta_b),
                    # (m + delta_m, b + delta_b),
                    # (m - delta_m, b - delta_b),
                    # (m - delta_m, b + delta_b),
                    # (m + delta_m, b - delta_b),
            ]
        mTmp, bTmp, scores, grid, max_score = score_grid(img, grid)
        if max_score >= current_score:
            m = mTmp
            b = bTmp
        # else:
        #     print ('Reached a peak?')
        delta_m *= delta_factor
        delta_b *= delta_factor
    printV('\tAccel. search end   m: {} b: {}'.format(m, b))
    return m, b    


def get_sigma_below(img, m, b):
    seg1, seg2 = split_img_by_line(img, m, b)
    return seg1.size / (seg1.size + seg2.size)


def optimize_scores(img, highres, slope_range, intercept_range, scaling_factor):
    """
        Params:
            img
        Returns:
            Answer: Tuple of (m, b)
            Scores (list of np.double)
            Grid
    """
    print(img.shape)
    printV('Optimize... img shape {} highres shape {}'.format(img.shape, highres.shape))
    grid = [(m, b) for b in intercept_range for m in slope_range]    

    print(len(grid))
    m, b, scores, grid, max_score = score_grid(img, grid)
    m2, b2 = accelerated_search(img, m, b * scaling_factor, max_score)
    b2 /= scaling_factor
    pitch, bank = convert_m_b_to_pitch_bank(m=m2, b=b2, sigma_below=get_sigma_below(img, m2, b2))
    print('\tPitch: {0:.2f}% \t Bank: {1:.2f} degrees'.format(pitch * 100, bank))
    return (m2, b2), scores, grid, pitch, bank


def optimize_global(img, highres, scaling_factor):
    printV('optimize_global()')
    return optimize_scores(img, highres,
                        slope_range=np.arange(-4, 4, 0.25), #0.25
                        intercept_range=np.arange( 1, img.shape[0] - 2, 0.1), #0.5
                        scaling_factor=scaling_factor)


def optimize_local(img, highres, m, b, scaling_factor):
    printV('optimize_local()')
    return optimize_scores(img, highres,
                        slope_range=np.arange(m - 0.5, m + 0.5, 0.2),
                        intercept_range=np.arange(max(1.0, b - 4.0), min(img.shape[0], b + 4.0), 1.0),
                        scaling_factor=scaling_factor)


def optimize_real_time(img, highres, m, b, scaling_factor):
    return (optimize_global(img, highres, scaling_factor) if not m or not b 
        else optimize_local(img, highres, m, b, scaling_factor))
