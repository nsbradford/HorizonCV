
import cv2
import numpy as np
from horizoncv.horizon import optimize_real_time, optimize_global, convert_m_b_to_pitch_bank
from . import plotter


def load_img(path):
    """ Load an image located at a path.
        Args:
            path (str): relative path to img
        Returns:
            resized (np.array): resized img
            img (np.array): unaltered loaded img
    """
    print ('load img...')
    img = cv2.imread(path)
    print('Image shape: ', img.shape) # rows, columns, depth (height x width x color)
    print('Resize...')
    resized = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
    # blur = cv2.GaussianBlur(resized,(3,3),0) # blurs the horizon too much
    print('Resized shape:', resized.shape, img.shape)
    return resized, img


def getVideoSource(filename):
    """ Generator for cleanly loading a video file (will only work if OpenCV was
            compiled with FFMPEG support). Videos are 1920 x 1080 original, 960 x 540 resized
    """ 
    print('Load video {}...'.format(filename))
    cap = cv2.VideoCapture('./media/' + filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            print (frame.shape)
            yield frame
    cap.release()


def optimization_surface_demo():
    """ Show the optimization surface in 2D and 3D views for a sample image. """
    #'taxi_rotate.png runway1.JPG taxi_empty.jpg ocean sunset grass
    img, full_res = load_img('./media/taxi_rotate.png')
    # img, full_res = load_img('../img/grass_pitch_low.jpg')  
    print('Optimize scores...')
    answer, scores, grid, pitch, bank = optimize_global(img, full_res, 1.0)
    # print('Max:', max_index)

    m = answer[0]
    b = answer[1]
    print('m:', m, '  b:', b)
    plotter.plot2D(full_res, scores, m, b*5, pitch, bank)

    X = [option[0] for option in grid] # m
    Y = [option[1] for option in grid] # b
    Z = list(map(lambda x: x * (10**8), scores))
    print(len(X), len(Y), len(Z))
    plotter.scatter3D(X, Y, Z) # scatter3D(X[::10], Y[::10], Z[::10])


def timer_demo():
    """ Time the LaneCV library to retrieve FPS (processing speed). """
    import timeit
    n_iterations = 1
    n_frames = 100
    result = timeit.timeit('demo.video_demo(filename="turn1.mp4", is_display=True, n_frames={})'.format(n_frames), 
                        setup='from horizoncv import demo;', 
                        number=n_iterations)
    seconds = result / n_iterations
    print('Timing: {} seconds for {} frames of video.'.format(seconds, n_frames))
    print('{} frames / second'.format(n_frames / seconds))


def video_demo(filename, is_display=True, n_frames=-1):
    """ Run LaneCV on a sample video. """
    highres_scale = 0.2
    scaling_factor = 0.1

    # cap = cv2.VideoCapture('./media/flying_turn.avi') #framerate of 25
    cap = cv2.VideoCapture() 
    count = 0
    m = None
    b = None
    for i, frame in enumerate(getVideoSource(filename)):
        print('---------------------\tFrame {}'.format(i+1))
        if n_frames > 0 and i > n_frames:
            break
        if count % 2 != 0:
            continue
        highres = cv2.resize(frame, dsize=None, fx=1.0 * highres_scale, fy=1.0 * highres_scale)
        img = cv2.resize(highres, dsize=None, fx=1.0 * scaling_factor, fy=1.0 * scaling_factor)
        answer, scores, grid, pitch, bank = optimize_real_time(img, img, m, b, 1.0) #scaling_factor)
        m, b = answer[0], answer[1]

        if is_display:
            # label = 'Prediction m: ' + str(m) + ' b: ' + str(b)
            prediction = plotter.add_line_to_frame(frame, m, b / (highres_scale * scaling_factor), pitch, bank)
            #m=np.float32(0.0), b=np.float32(30.0))
            cv2.imshow('frame',prediction)
            # cv2.imshow('label', cv2.resize(prediction, dsize=None, fx=0.3, fy=0.3))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
1