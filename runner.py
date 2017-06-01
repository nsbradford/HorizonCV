"""
    runner.py
    Nicholas S. Bradford
    4/12/2017

"""

import nose
import cv2
import numpy as np

from horizoncv import demo


def testAll():
    print('Test...')
    argv = ['fake', 
            '-verbosity=2', 
            '--nocapture', 
            '--with-coverage', 
            '--cover-package=horizoncv']
    result = nose.run(argv=argv)
    return result


if __name__ == '__main__':
    testResult = testAll()
    if testResult:
        # demo.time_score()
        # demo.optimization_surface_demo()
        # demo.timerDemo() # framerate of 25
        # demo.video_demo('flying_turn.avi')
        demo.video_demo('turn1.mp4')
