

'''
This is a full Ratslam implementation in python. This implementation is based 
on Milford's original implementation [1]_ in matlab, and Christine Lee's python 
implementation [2]_. The original data movies can also be found in [1]_.

The only dependence for this package is Numpy [3]_, thus it does not handle how
to open and manage the movie and image files. For this, I strongly recommend 
the use of OpenCV [4]_.

.. [1] https://wiki.qut.edu.au/display/cyphy/RatSLAM+MATLAB
.. [2] https://github.com/coxlab/ratslam-python
.. [3] http://www.numpy.org/
.. [4] http://opencv.org/
'''

import numpy as np
from ratslam._globals import *
from ratslam.visual_odometry import VisualOdometry
from ratslam.view_cells import ViewCells
from ratslam.pose_cells import PoseCells
from ratslam.experience_map import ExperienceMap

class Ratslam(object):
    '''Ratslam implementation.

    The ratslam is divided into 4 modules: visual odometry, view cells, pose 
    cells, and experience map. This class also store the odometry and pose 
    cells activation in order to plot them.
    '''

    def __init__(self):
        '''Initializes the ratslam modules.'''

        self.visual_odometry = VisualOdometry()
        self.view_cells = ViewCells()
        self.pose_cells = PoseCells()
        self.experience_map = ExperienceMap()

        # TRACKING -------------------------------
        x, y, th = self.visual_odometry.odometry
        self.odometry = [[x], [y], [th]]
        
        x_pc, y_pc, th_pc = self.pose_cells.active
        self.pc = [[x_pc], [y_pc], [th_pc]]
        # ----------------------------------------

    def digest(self, img):
        '''Execute a step of ratslam algorithm for a given image.

        :param img: an gray-scale image as a 2D numpy array.
        '''

        x_pc, y_pc, th_pc = self.pose_cells.active
        view_cell = self.view_cells(img, x_pc, y_pc, th_pc)
        vtrans, vrot = self.visual_odometry(img)
        x_pc, y_pc, th_pc = self.pose_cells(view_cell, vtrans, vrot)
        self.experience_map(view_cell, vtrans, vrot, x_pc, y_pc, th_pc)

        # TRACKING -------------------------------
        x, y, th = self.visual_odometry.odometry
        self.odometry[0].append(x)
        self.odometry[1].append(y)
        self.odometry[2].append(th)
        self.pc[0].append(x_pc)
        self.pc[1].append(y_pc)
        self.pc[2].append(th_pc)
        # ----------------------------------------
