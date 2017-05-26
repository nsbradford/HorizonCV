

# def print_results(m, b, m2, b2):
#     print('\tInitial answer - m:', m, '  b:', b)
#     print('\tAccelerate search...')
#     print('\tRefined_answer: - m:', m2, '  b:', b2)

# scores = list(map(lambda x: score_line(img, x[0], x[1]), grid))
# assert len(scores) > 0, 'Invalid slope and intercept ranges: ' + str(slope_range) + str(intercept_range)
# max_index = np.argmax(scores)
# m, b = grid[max_index]


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