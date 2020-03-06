# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Feb 19 08:51:41 2020

@author: ixtiyor
"""
from random import randrange
import warnings
import numpy as np
import cv2
import os
import math
#from common import anorm2, draw_str
from pclines_point_alignment import params, detect_vps_given_lines
# parameters to change
from pc_lines_diamond.diamond_vanish import diamond_vanish
class App:
    def __init__(self, video_src):
        # parameters to change
        self.track_len = 6
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.track_circle_color = (0, 255, 0)
        self.track_line_color = (0, 0, 255)
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
        self.track_lines = []
        self.track_lines_coeffs = []
        self.feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )
        self.track_len_threshold = self.track_len * 0.3
        self.points_staright_old = []
        self.points_twisted_old = []
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
    
    def get_len_track(self, track):
        x = track[0][0]
        y = track[0][1]
        len_= 0
        diff_xs = 0
        diff_ys = 0
        for j in np.arange(1, len(track),1):
            xn = track[j][0]
            yn = track[j][1]
            diff_x = (xn-x)
            diff_y = (yn-y)
            len_ += math.sqrt(diff_x * diff_x + diff_y * diff_y)
            diff_xs += diff_x
            diff_ys += diff_y
        
        return len_

    def run(self):
        _ret, frame = self.cam.read()
        frame = cv2.resize(frame, (512,512 ))
        width = frame.shape[1]
        height = frame.shape[0]
        self.prms = params(width, height)
        vis = frame.copy()
        while True:
            _ret, frame = self.cam.read()
            frame = cv2.resize(frame, (width,height ))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #vis = np.zeros(frame.shape, np.uint8)                                                                        
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1 # parameter to change
                new_tracks = []  
                new_lines = []
                new_line_coeffs = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
#                    print("(x, y)", (x, y))
                    tr.append((x, y))
                    if len(tr) > self.track_len:

                        del tr[0]
                    if(self.get_len_track(tr) > self.track_len_threshold):
                        my_tr = np.array(tr)                    
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', np.RankWarning)
                            p = np.poly1d(np.polyfit(my_tr[:,0],my_tr[:,1],1))
#                            print('polyfitted', p.)
                            # make line that goes to infinity, in our case it is image boundary
    #                        print('line eq', p)
    #                        print('coefficients', np.polyfit(my_tr[:,0],my_tr[:,1], 1))..
    #                        print('line',[0, p(0),511, p(-511)])
                            new_lines.append([0, p(0), 511, p(511)])
                            # a, b, c, w
                            # p[0], 1, p[1], 1
                            new_line_coeffs.append([p.c[0], -1, p.c[1], 1])
#                            cv2.line( vis, (0, int(p(0))), (511, int(p(511))), (255,0,122), 2, 8 );
                    new_tracks.append(tr)
                    
                    if self.frame_idx % self.detect_interval == 0:
                        vis = frame.copy()
                        cv2.circle(vis, (x, y), 1, self.track_circle_color, -1)
                self.tracks = new_tracks
                self.track_lines = new_lines
                self.track_lines_coeffs = new_line_coeffs
                
#            print("self.traks=", np.shape(self.tracks))
            if self.frame_idx % self.detect_interval == 0:
                if(len(self.track_lines)>0):
#                    _, self.points_staright_old,self.points_twisted_old =  detect_vps_given_lines(frame_gray,self.prms,np.array(self.track_lines), vis,self.points_staright_old,self.points_twisted_old)
                    
                    result = diamond_vanish(np.array(self.track_lines_coeffs), 0.4, 100,1, [self.prms.h,self.prms.w])
                    resvps = np.int32(abs(result["CC_VanP"]))
                    print("detected vp =====", resvps)
                    for i in range(len(resvps)):
                        cv2.circle(vis, (resvps[i][0], resvps[i][1]), 5, self.track_circle_color, -1)
                    
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 2, 0, -1)
                
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

                for i in range(len(self.tracks)):
                     if(self.get_len_track(self.tracks[i]) > self.track_len_threshold):
                        cv2.polylines(vis, np.int32([self.tracks[i]]), False, self.track_line_color)

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            ch = cv2.waitKey(1)
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