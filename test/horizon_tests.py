import unittest
import numpy as np

from horizoncv import horizon

class HorizonTest(unittest.TestCase):

    def test_split_img_by_line_simple(self):
        img = np.ones((8, 8, 3))
        m = 0.0
        b = 4.0
        seg1, seg2 = horizon.split_img_by_line(img, m, b)
        self.assertEquals((8 * 8 / 2, 3), seg1.shape)
        self.assertEquals((8 * 8 / 2, 3), seg2.shape)

    def test_split_img_by_line_simple(self):
        img = np.ones((8, 8, 3))
        m = 1.0
        b = 0.0
        seg1, seg2 = horizon.split_img_by_line(img, m, b)
        self.assertEquals((8 * 8 / 2 + 4, 3), seg1.shape)
        self.assertEquals((8 * 8 / 2 - 4, 3), seg2.shape)

    def test_img_line_mask(self):
        rows = 15
        columns = 24
        slope_range = list(np.arange(-4, 4, 0.25))
        intercept_range = list(np.arange( 1, rows - 2, 0.5))

        # TODO
        # for m, b in zip(slope_range, intercept_range):
        #     print(m, b)
        #     mask1 = horizon.img_line_mask(rows, columns, m, b)
        #     mask2 = horizon.img_line_mask2(rows, columns, m, b)
        #     self.assertTrue(np.array_equal(mask1, mask2), str(mask1) + '2' + str(mask2))