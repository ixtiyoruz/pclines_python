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
from pc_lines_diamond.mx_lines import  fit_ellipse, gety,get_coeffs
#from pclines import params, denoise_lanes
#parameters to change
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
def twoPoints2Polar(line):
    p1 = np.array(line[0])
    p2 = np.array(line[1])
    # Compute 'rho' and 'theta'
    rho = abs(p2[0]*p1[1] - p2[1]*p1[0]) / cv.norm(p2 - p1);
    theta = -np.arctan2((p2[0] - p1[0]) , (p2[1] - p1[1]));

    # You can have a negative distance from the center 
    # when the angle is negative
    if (theta < 0):
        rho = -rho;

    return rho, theta
class App:
    def __init__(self, video_src):
        # parameters to change
        self.track_len = 4
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0
    
    def get_len_tracks(self, i):
        x = self.tracks[i][0][0]
        y = self.tracks[i][0][1]
        len_= 0
        diff_xs = 0
        diff_ys = 0
        for j in np.arange(1, len(self.tracks[i]),1):
            xn = self.tracks[i][j][0]
            yn = self.tracks[i][j][1]
            diff_x = (xn-x)
            diff_y = (yn-y)
            len_ += math.sqrt(diff_x * diff_x + diff_y * diff_y)
            diff_xs += diff_x
            diff_ys += diff_y
        
        return len_, (diff_xs, diff_ys)
    
    def line_intersection_point(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
    
        div = det(xdiff, ydiff)
        if div == 0:
           return None, None
    
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    
    def take_n_best_lanes(self, accumulator, thetas, rhos, n = 3):
        rho_id, t_id = np.unravel_index(np.argmin(accumulator), np.array(accumulator).shape)
        min_value = accumulator[rho_id, t_id]
        accumulator_temp = accumulator.copy()
        points = []
        for i in range(n):
            rho_id, t_id = np.unravel_index(np.argmax(accumulator_temp), np.array(accumulator_temp).shape)
            accumulator_temp[rho_id, t_id] = min_value
            rho = rhos[rho_id]
            theta = thetas[t_id]
            a, b= math.cos(theta), math.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pt1 = (int(round(x0 + 1000*(-b))), int(round(y0 + 1000*(a))))
            pt2 = (int(round(x0 - 1000*(-b))), int(round(y0 - 1000*(a))))
            points.append((pt1, pt2))
        return points
    
    def take_lane_towards_horizon(self,point1, point2, length=30):
        xdiff = (point1[0] - point2[0])
        ydiff = (point1[1] - point2[1])
        len_ = math.sqrt(xdiff*xdiff + ydiff*ydiff)
        return (point1, (int(point1[0] + length* xdiff/len_),int(point1[1] + length* ydiff/len_)))
    
    def run(self):
        thetas = np.deg2rad(np.arange(-90.0, 90.0))
        num_thetas = len(thetas)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        _ret, frame = self.cam.read()
        frame = cv.resize(frame, (512,512 ))
        width = frame.shape[1]
        height = frame.shape[0]
        
#        self.prms = params(width, height)

        diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
        diag_len = int(diag_len)
        accumulator1 = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
        accumulator2 = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
        while True:
            _ret, frame = self.cam.read()
            frame = cv.resize(frame, (width,height ))
            rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            #vis = np.zeros(frame.shape, np.uint8)                                                                        
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1 # parameter to change
                new_tracks = []                                                                                                                                                                                                                                                                         
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 1, (0, 255, 0), -1)
                self.tracks = new_tracks
                
                print('track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        
            #     lines = lsd.lsd(frame_gray)
            # for i in range(lines.shape[0]):
            #     pt1 = (int(lines[i, 0]), int(lines[i, 1]))
            #     pt2 = (int(lines[i, 2]), int(lines[i, 3]))
            #     cv.line(vis, pt1,pt2, (0, 0, 255), 1)
            
#            denoised_lanes = denoise_lanes(self.tracks, self.prms)

#            print(np.shape(denoised_lanes), len(self.tracks),self.prms.LENGTH_THRESHOLD)
            for i in range(len(self.tracks)):
                len_, (diff_xs, diff_ys) = self.get_len_tracks(i)
                if(len_ < 30): continue
                if(diff_ys < 0):
                    cv.polylines(vis, np.int32([self.tracks[i]]), False, (0, 0, 255))
                else:
                    cv.polylines(vis, np.int32([self.tracks[i]]), False, (255, 0, 0))
#                for j in range(len(self.tracks[i])):
#                    for t_idx in np.arange(0,num_thetas, 5):
#                print(np.shape(self.tracks[i]))
                a,b,c = get_coeffs(np.array(self.tracks[i]))
                xs = [-width,width-1]
                ys= [gety(xs[0], a,b,c),gety(xs[1], a,b,c)]
                rho, theta = twoPoints2Polar([[xs[0], ys[0]], [xs[1], ys[1]]])
#                        m = (ys[1] - ys[0]) / (xs[1] - xs[0])
#                        theta = np.arctan2()
#                        x = self.tracks[i][j][0]
#                        y = self.tracks[i][j][1]
                
                # Calculate rho. diag_len is added for a positive index
#                        rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
                rho = rho + diag_len
                print(rho, theta)
                if(diff_ys < 0):
                    accumulator1[int(rho), int(theta)] += 1
                else:
                    accumulator2[int(rho), int(theta)] += 1
                            
            
#            hough_image = draw_hough_line(accumulator1,thetas, rhos )
#            hough_image2 = draw_hough_line(accumulator2,thetas, rhos )
            
            points_acc1 = self.take_n_best_lanes(accumulator1, thetas, rhos, n = 1)
            for i in range(len(points_acc1)):
                cv.line( vis, points_acc1[i][0], points_acc1[i][1], (122,0,255), 2, 8 );
            points_acc2 = self.take_n_best_lanes(accumulator2, thetas, rhos, n = 1)
            for i in range(len(points_acc1)):
                cv.line( vis, points_acc2[i][0], points_acc2[i][1], (255,0,122), 2, 8 );

            x_int, y_int= self.line_intersection_point(points_acc1[0], points_acc2[0])
            if(not x_int == None):
                cv.circle(vis, (int(x_int), int(y_int)), 3, (0, 255, 0), -1)
                for i in range(10):
                    random.seed(i)
                    psample1 = self.take_lane_towards_horizon((randrange(width), randrange(height)), (x_int, y_int), length=30)
                    cv.arrowedLine(	vis,psample1[1],  psample1[0], (122,255,255), 2, 8)
            
#            print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)
#            print(np.shape(vis), np.shape(frame), type(vis), type(vis[0,0,0]), type(frame[0,0,0]))
#            blnd_img = cv.add(vis,frame)
#            cv.imshow('img', blnd_img)
#            cv.imshow('hough_img1', hough_image)
#            cv.imshow('hough_img2', hough_image2)
            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
#    videos_root = '/media/ixtiyor/New Volume/datasets/auto_callibration/d1'
#    videos = os.listdir(videos_root)
    try:
        video_src = "GOPR2036_half.mp4" #videos_root + "/" + videos[10]
    except:
        video_src = 0

    App(video_src).run()
    print('Done')
if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
