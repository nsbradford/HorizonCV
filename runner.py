
import cv2 # for reading photos and videos
import numpy as np
from horizoncv.horizon import optimize_real_time, optimize_global, convert_m_b_to_pitch_bank
from horizoncv import plotter


def load_img(path):
    print ('load img...') # taxi_rotate.png
    img = cv2.imread(path)
    print('Image shape: ', img.shape) # rows, columns, depth (height x width x color)
    print('Resize...')
    resized = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
    # blur = cv2.GaussianBlur(resized,(3,3),0) # blurs the horizon too much
    print('Resized shape:', resized.shape, img.shape)
    return resized, img


def basic_test():
    img = load_img('../img/taxi_rotate.png') #'../img/runway1.JPG' taxi_empty.jpg ocean sunset grass
    good_line = score_line(img, m=0.0, b=20)
    bad_line = score_line(img, m=2.0, b=0)
    assert good_line > bad_line
    print('Basic test of scoring...')


def time_score():
    import timeit
    result = timeit.timeit('horizon.score_line(img, m=0.0,  b=20)', 
                        setup='import horizon; img=horizon.load_img();', 
                        number=1000)
    print('Timing:', result/1000, 'seconds to score a single line.')


def main():
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


def video_demo():
    highres_scale = 0.5
    scaling_factor = 0.2

    print('Load video...')
    cap = cv2.VideoCapture('./media/flying_turn.avi') #framerate of 25
    count = 0
    m = None
    b = None
    while(cap.isOpened()):
        # print('Read new frame...')
        ret, frame = cap.read()
        count += 1
        if count % 5 != 0:
            continue
        if not ret:
            break
        highres = cv2.resize(frame, dsize=None, fx=1.0 * highres_scale, fy=1.0 * highres_scale)
        img = cv2.resize(highres, dsize=None, fx=1.0 * scaling_factor, fy=1.0 * scaling_factor)

        answer, scores, grid, pitch, bank = optimize_real_time(img, img, m, b, 1.0) #scaling_factor)
        m = answer[0]
        b = answer[1]
        # label = 'Prediction m: ' + str(m) + ' b: ' + str(b)
        prediction = plotter.add_line_to_frame(frame, m, b / (highres_scale * scaling_factor), pitch, bank)
        #m=np.float32(0.0), b=np.float32(30.0))
        
        # cv2.imshow('frame',frame)
        
        cv2.imshow('label', prediction)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # time_score()2
    main()
    video_demo()