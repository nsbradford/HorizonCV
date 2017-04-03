
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def add_line_to_frame(img, m, b, pitch, bank):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pt1 = (0, b.astype(np.int64))
    pt2 = img.shape[1] - 1, (m * (img.shape[1] - 1) + b).astype(np.int64)
    cv2.line(img=rgb_img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = 'Pitch: ' + "{:.1f}".format(pitch) + ' %     Bank: ' + "{:.1f}".format(bank) + ' degrees'
    cv2.putText(rgb_img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return rgb_img


def plot2D(img, scores, m, b, pitch, bank):
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = 'Pitch: ' + "{:.1f}".format(pitch) + ' %     Bank: ' + "{:.1f}".format(bank) + ' degrees'
    cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    pt1 = (0, b.astype(np.int64))
    pt2 = img.shape[1] - 1, (m * (img.shape[1] - 1) + b).astype(np.int64)
    cv2.line(img=img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Input')
    
    plt.subplot(122)
    y = list(map(lambda x: x * (10**8), scores))
    x = list(range(len(scores)))
    plt.fill_between(x, y, [0 for x in range(len(scores))], color='grey')
    # plt.imshow(resized),plt.title('Output')
    plt.show()


def scatter3D(X, Y, Z):
    print('Plot in 3D...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=np.abs(Z), cmap=cm.coolwarm)
    ax.set_xlabel('M (slope)')
    ax.set_ylabel('B (intercept)')
    ax.set_zlabel('Z Label')
    plt.show()
