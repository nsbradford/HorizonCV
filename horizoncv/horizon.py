""" 
    horizon.py
    Nicholas S. Bradford

    Use algorithm from "Vision-guided flight stability and control for micro
        air vehicles" (Ettinger et al. ).
    Intuition: horizon will be a line dividing image into two segments with low variance,
        which can be modeled as minimizing the product of the three eigenvalues of the
        covariance matrix (the determinant). 

    In its current form, can run 10000 iterations in 9.1 seconds, or about 30 per iteration
        at 30Hz. Java performance benefit of 10x would mean 300 per iteration,
        and moving to 10Hz would leave ~1000 per iteration.
    The initial optimized search grid is on a 12x12 grid (144 values),
        which is then refined on a full-resolution image using a gradient-descent-like
        sampling technique. (requires 4 checks at each step and ~7 steps = ~28, but
        will but at higher resolution)
    Total requirements: must be able to run at least ~200 checks/second

"""

import cv2
import numpy as np
import math


def convert_m_b_to_pitch_bank(m, b, sigma_below):
    """ 'The pitch angle cannot be exactly calculated from an arbitrary horizon line, however
            the pitch angle will be closely proportional to the percentage of the image above 
            or below the line.'
        Pitch angle (Theta) = size(ground) / size(ground) + size(sky)
        Bank angle (Phi) = tan^-1(m)
    """

    bank = math.degrees(math.atan(m))
    
    # method from original paper
    pitch = sigma_below

    # method from 
    # 'Road environment modeling using robust perspective analysis and recursive Bayesian segmentation'
    # focal_x = 64.4 #
    # focal_y = 37.2 #
    # s = 0
    # x0 = 25 / 2 # image center, in pix
    # y0 = 52 / 2 # image center, in pix
    # cam_calibration = np.array([[focal_x, s, x0], 
    #                             [0, focal_y, y0],
    #                             [0, 0, 1]])
    # v = np.array([x0, b, 1])
    # v_prime = np.linalg.solve(cam_calibration, v)
    # print(v_prime, b)
    # pitch = math.atan(v[1])

    # method from http://eprints.qut.edu.au/12839/1/3067a485.pdf
    # u = 5
    # v = m * u + b
    # f = 35.0
    # inner = (u * math.sin(bank) + v * math.sin(bank)) / f
    # # print(math.atan(inner), math.atan(- inner), math.atan(inner) - math.atan(- inner))
    # pitch = math.atan(inner)
    # print(inner, pitch)

    # method from https://www.researchgate.net/publication/
    #   220143231_Sub-sampling_Real-time_vision_for_micro_air_vehicles
    # images are 29 x 35, or 36 x 64
    # h = 29 # height of image
    # w = 35.0
    # h = 20
    # w = 35
    # y = m * (w / 2.0) + b # y-coordinate of line at half of the screen
    # FOVv = 40 # camera's vertical field of view
    # pitch = y * FOVv / h

    return pitch, bank


def convert_pitch_roll_to_m_b(pitch, bank):
    """ 
        Limits of (pitch, roll) space are [-pi/2, pi/2] for pitch and [0%, 100%] for roll
    """
    m = math.tan(pitch)
    b = None
    return (m, b)


def img_line_mask(rows, columns, m, b):
    """ Params:
            rows (int)
            columns (int)
            m (double)
            b (double)
        Returns:
            rows x columns np.array boolean mask with True for all values above the line
    """
    # TODO: there must be a way to optimize this
    mask = np.zeros((rows, columns), dtype=np.bool)
    for y in range(rows):
        for x in range(columns):
            if y > m * x + b:
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

        When the covariance matrix is nearly singular (due to color issues), the determinant
        will also be driven to zero. Thus, we introduce additional terms to supplement the
        score when this case occurs (the determinant dominates it in the normal case):
          where g=GROUND and s=SKY (covariance matrices) 
          F = [det(G) + det(S) + (eigG1 + eigG1 + eigG1)^2 + (eigS1 + eigS1 + eigS1)^2]^-1
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
    for i in range(max_iter):
        # print('\tDelta', delta_m, delta_b, 'M&B', m, b)
        grid = [
                    (m + delta_m, b),
                    (m - delta_m, b),
                    (m, b + delta_b),
                    (m, b - delta_b),
                    (m + delta_m, b + delta_b),
                    (m - delta_m, b - delta_b),
                    (m - delta_m, b + delta_b),
                    (m + delta_m, b - delta_b),
            ]
        mTmp, bTmp, scores, grid, max_score = score_grid(img, grid)
        if max_score >= current_score:
            m = mTmp
            b = bTmp
        # else:
        #     print ('Reached a peak?')
        delta_m *= delta_factor
        delta_b *= delta_factor
    return m, b    



def get_simga_below(img, m, b):
    seg1, seg2 = split_img_by_line(img, m, b)
    return seg1.size / (seg1.size + seg2.size)

def print_results(m, b, m2, b2):
    print('\tInitial answer - m:', m, '  b:', b)
    print('\tAccelerate search...')
    print('\tRefined_answer: - m:', m2, '  b:', b2)

def optimize_scores(img, highres, slope_range, intercept_range, scaling_factor):
    """
        Params:
            img
        Returns:
            Answer: Tuple of (m, b)
            Scores (list of np.double)
            Grid
    """
    print('Optimize...', img.shape, highres.shape)
    grid = [(m, b) for b in intercept_range for m in slope_range]    
    # scores = list(map(lambda x: score_line(img, x[0], x[1]), grid))
    # assert len(scores) > 0, 'Invalid slope and intercept ranges: ' + str(slope_range) + str(intercept_range)
    # max_index = np.argmax(scores)
    # m, b = grid[max_index]
    m, b, scores, grid, max_score = score_grid(img, grid)
    m2, b2 = accelerated_search(img, m, b * scaling_factor, max_score)
    b2 /= scaling_factor
    pitch, bank = convert_m_b_to_pitch_bank(m=m2, b=b2, sigma_below=get_simga_below(img, m2, b2))
    # print('\tPitch:', pitch, '%  Bank:', bank, 'degrees')
    return (m2, b2), scores, grid, pitch, bank


def optimize_global(img, highres, scaling_factor):
    return optimize_scores(img, highres,
                        slope_range=np.arange(-4, 4, 0.25),
                        intercept_range=np.arange( 1, img.shape[0] - 2, 0.5),
                        scaling_factor=scaling_factor)

def optimize_local(img, highres, m, b, scaling_factor):
    return optimize_scores(img, highres,
                        slope_range=np.arange(m - 0.5, m + 0.5, 0.05),
                        intercept_range=np.arange(max(1.0, b - 4.0), min(img.shape[0], b + 4.0), 0.5),
                        scaling_factor=scaling_factor)

def optimize_real_time(img, highres, m, b, scaling_factor):
    return (optimize_global(img, highres, scaling_factor) if not m or not b 
        else optimize_local(img, highres, m, b, scaling_factor))
