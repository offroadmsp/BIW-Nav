

import cv2
import numpy as np
import ratslam

'''
Utility for profiling.

Use it with RunSnakeRun [1]_ and Line Profiler [2]_.

**RunSnakeRun:**

 - For RunSnakeRun, just run this file and open the ratslam.profile file into 
   the program. I suggest you to backup the profile file in order to not 
   override it when running the line profiler.

**Line Profiler:**

 - For Line Profiler, first insert the @profile decoration into some point of
   the code, put a break point into the main loop if needed, and finally, run::

    $ kernprof.py -l profiling.py && python -m line_profiler profiling.py.lprof 


.. [1] http://www.vrplumber.com/programming/runsnakerun/
.. [2] http://pythonhosted.org/line_profiler/
'''

def main():
    data = r'D:\Bkp\ratslam\data\stlucia_testloop.avi'

    video = cv2.VideoCapture(data)
    slam = ratslam.Ratslam()
    
    loop = 0
    _, frame = video.read()
    while True:
        loop += 1
        _, frame = video.read()
        if frame is None: break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        slam.digest(img)
        
        # break

        for loop in range(1001):
            if loop % 100 == 0:
                print('{:0.2f}%'.format(100 * loop / 1000.))

import cProfile
command = """main()"""
cProfile.runctx(command, globals(), locals(), filename="ratslam.profile" )