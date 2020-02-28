# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Feb 19 08:51:41 2020

@author: ixtiyor
"""
from random import randrange

import numpy as np
import cv2 as cv
import os
import math
import random
#from common import anorm2, draw_str
from pylsd import lsd
import matplotlib.pyplot as plt
from pclines import params, denoise_lanes, convert_to_PClines, detect_vps
# parameters to change
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )
def draw_hough_line(accumulator, thetas, rhos):
    fig, ax0 = plt.subplots(nrows=1, figsize=(10, 10))


    ax0.imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_title('Hough transform')
    ax0.set_xlabel('Angles (degrees)')
    ax0.set_ylabel('Distance (pixels)')
#    ax0.axis('image')
    fig.tight_layout()
    fig.canvas.draw()
#    X = np.array(fig.canvas.renderer.buffer_rgba())
    
    X = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    X = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.canvas.flush_events()
    fig.clf()
#    plt.show()
    return X

class App:
    def __init__(self, video_src):
        # parameters to change
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0
        self.detect_interval = 5

    def run(self):
        _ret, frame = self.cam.read()
        frame = cv.resize(frame, (512,512 ))
        width = frame.shape[1]
        height = frame.shape[0]
        
        self.prms = params(width, height)

        while True:
            _ret, frame = self.cam.read()
            frame = cv.resize(frame, (width,height ))
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            if self.frame_idx % self.detect_interval == 0:
                detect_vps(frame_gray, self.prms, vis)
                
#                lines = lsd.lsd(np.array(frame_gray, np.float32))
#                lines = lines[:,0:4]
#            for i in range(lines.shape[0]):
#                pt1 = (int(lines[i, 0]), int(lines[i, 1]))
#                pt2 = (int(lines[i, 2]), int(lines[i, 3]))
#                cv.line(vis, pt1,pt2, (0, 0, 255), 1)
#            denoised_lanes = denoise_lanes(lines, self.prms)
#            if(len(lines) > 0):
#                points_staright, points_twisted = convert_to_PClines(denoised_lanes, self.prms)
#                print(np.shape(points_staright), np.shape(points_twisted))
#            self.frame_idx += 1
#            self.prev_gray = frame_gray
                
                
            cv.imshow('lk_track', vis)
            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    videos_root = '/media/ixtiyor/New Volume/datasets/auto_callibration/d1'
    videos = os.listdir(videos_root)
    try:
        video_src = videos_root + "/" + videos[1]
    except:
        video_src = 0

    App(video_src).run()
    print('Done')
    
if __name__ == "__main__":
     main()  
"""
import cartesius.main as cartesius
import cartesius.elements as elements
import cartesius.charts as charts
import math 
coordinate_system = cartesius.CoordinateSystem(bounds=(-10, 10, -10, 10))

f = lambda x : math.sin(x) * 2
#    coordinate_system.add(charts.Function(f, start=-4, end=5, step=0.02, color=0x0000ff))
point1 = (3, -3)
point2 = (-1, 8)
coordinate_system.add(elements.Circle(point1, radius=0.2,fill_color = (255, 0, 0), color =  (255, 0, 0)))
coordinate_system.add(elements.Circle(point2, radius=0.2,fill_color = (255, 0, 0), color =  (255, 0, 0)))
coordinate_system.add(elements.Line(point1, point2))
coordinate_system.add(elements.Point(point1,  style='x', label='A'))
coordinate_system.add(elements.Point(point2, style='x', label='B'))

coordinate_system.add(elements.Axis(horizontal=True, points=1, labels=1, label_position=cartesius.LEFT_UP,))
coordinate_system.add(elements.Axis(vertical=True, points=1, labels=1, label_position=cartesius.LEFT_CENTER,))



#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#vectors to plot: 4D for this example
y1=[3,-3]

#y2=[1.5,1.7,2.2,2.9]

x=[1,2] # spines

#fig,(ax,ax2,ax3) = plt.subplots(1, 1, sharey=False)
fig,(ax) = plt.subplots(1, 1, sharey=False)

# plot the same on all the subplots
ax.plot(x,np.array(point1)/10.0,'r-')
ax.plot(x,np.array(point2)/10.0,'r-')
#ax2.plot(x,y1,'r-')
#ax3.plot(x,y1,'r-')

# now zoom in each of the subplots 
ax.set_xlim([ x[0],x[1]])
#ax2.set_xlim([ x[1],x[2]])
#ax3.set_xlim([ x[2],x[3]])

# set the x axis ticks 
for axx,xx in zip([ax],x[:-1]):
  axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
#ax3.xaxis.set_major_locator(ticker.FixedLocator([x[-2],x[-1]]))  # the last one

# EDIT: add the labels to the rightmost spine
for tick in ax.yaxis.get_major_ticks():
  tick.label2On=True

# stack the subplots together
plt.subplots_adjust(wspace=0)

plt.show()
coordinate_system.draw(800, 640)
from pclines import PCLines_straight_all
prms = params(10, 10)
PCLines_straight_all(np.expand_dims([310, 330, 10,303],0)/480.0)
PCLines_straight_all(np.expand_dims(point2,0)/10.0)
"""