

import numpy as np
from ratslam._globals import *

class ViewCell(object):
    '''A single view cell.

    A ViewCell object is used to store the information of a single view cell.
    '''
    _ID = 0

    def __init__(self, template, x_pc, y_pc, th_pc):
        '''Initialize a ViewCell.

        :param template: a 1D numpy array with the cell template.
        :param x_pc: the x position relative to the pose cell.
        :param y_pc: the y position relative to the pose cell.
        :param th_pc: the th position relative to the pose cell.
        '''
        self.id = ViewCell._ID
        self.template = template
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.decay = VT_ACTIVE_DECAY
        self.first = True
        self.exps = []

        ViewCell._ID += 1

class ViewCells(object):
    '''View Cell module.'''

    def __init__(self):
        '''Initializes the View Cell module.'''
        self.size = 0
        self.cells = []
        self.prev_cell = None

        # self.create_cell(np.zeros(561), 30, 30, 18)

    def _create_template(self, img):
        '''Compute the sum of columns in subimg and normalize it.

        :param subimg: a sub-image as a 2D numpy array.
        :return: the view template as a 1D numpy array.
        '''
        # print(img.shape)
        # print(IMAGE_VT_Y_RANGE)
        # print(IMAGE_VT_X_RANGE)
        subimg = img[IMAGE_VT_Y_RANGE, IMAGE_VT_X_RANGE]
        x_sums = np.sum(subimg, 0)
        return x_sums/np.sum(x_sums, dtype=np.float32)

    def _score(self, template):
        '''Compute the similarity of a given template with all view cells.

        :param template: 1D numpy array.
        :return: 1D numpy array.
        '''
        scores = []
        for cell in self.cells:

            cell.decay -= VT_GLOBAL_DECAY
            if cell.decay < 0:
                cell.decay = 0

            _, s = compare_segments(template, cell.template, VT_SHIFT_MATCH)
            scores.append(s)

        return scores

    def create_cell(self, template, x_pc, y_pc, th_pc):
        '''Create a new View Cell and register it into the View Cell module

        :param template: 1D numpy array.
        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :return: the new View Cell.
        '''
        cell = ViewCell(template, x_pc, y_pc, th_pc)
        self.cells.append(cell)
        self.size += 1
        return cell

    def __call__(self, img, x_pc, y_pc, th_pc):
        '''Execute an interation of visual template.

        :param img: the full gray-scaled image as a 2D numpy array.
        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :return: the active view cell.
        '''
        template = self._create_template(img)
        scores = self._score(template)

        # TO REMOVE
        if scores:
            self.activated_cells = np.array(self.cells)[np.array(scores)*template.size<VT_MATCH_THRESHOLD]
        # ----

        if not self.size or np.min(scores)*template.size > VT_MATCH_THRESHOLD:
            cell = self.create_cell(template, x_pc, y_pc, th_pc)
            self.prev_cell = cell
            return cell

        i = np.argmin(scores)
        cell = self.cells[i]
        cell.decay += VT_ACTIVE_DECAY

        if self.prev_cell != cell:
            cell.first = False

        self.prev_cell = cell
        return cell
