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
from pylsd import lsd
import os
import math
#from common import anorm2, draw_str
from pclines_point_alignment import params#, detect_vps_given_lines
# parameters to change
from pc_lines_diamond.diamond_vanish import diamond_vanish, normalize_PC_points, diamond_vanish_with_lines
from pc_lines_diamond.ransac.ransac import run_ransac, estimate, is_inlier
from pc_lines_diamond.utils import get_diamond_c_from_original_coords, get_original_y_from_diamond_space, gety, get_original_c_from_original_points
from pc_lines_diamond.diamond_vanish import diamond_vanish, normalize_PC_points, diamond_vanish_with_lines
from pc_lines_diamond.mx_lines import  fit_ellipse, gety,get_coeffs
from moving_edge_main import background_test, get_orientation_matrix_way
import random

class App:
    def __init__(self, video_src):
        # parameters to change
        self.track_len = 6
        self.alpha = 0.95 # for moving edge detection
        self.B = [] # for moving edge detection, edge bins
        self.detect_interval = 3
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
        self.moving_line_coeffs = []
        self.feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )
#        self.track_len_threshold = self.track_len * 0.3
        self.space_old = []
        self.space_old_moving_edges = [] # for second vanishing point
        self.vp_1 = [] # first vanishing pooint
        self.vp_2 = [] # second vanishing pooint
    def take_lane_towards_horizon(self,point1, length=30):
        xdiff_1 = (point1[0] - self.vp_1[0])
        ydiff_1 = (point1[1] - self.vp_1[1])
        len_1 = math.sqrt(xdiff_1*xdiff_1 + ydiff_1*ydiff_1)
        line1 = [point1, (int(point1[0] + length* xdiff_1/len_1),int(point1[1] + length* ydiff_1/len_1))]
        
        xdiff_2 = (point1[0] - self.vp_2[0])
        ydiff_2 = (point1[1] - self.vp_2[1])
        len_2 = math.sqrt(xdiff_2*xdiff_2 + ydiff_2*ydiff_2)
        line2 = [point1, (int(point1[0] + length* xdiff_2/len_2),int(point1[1] + length* ydiff_2/len_2))]
        
        return line1, line2
    
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
                
#        frame = np.zeros((400, 400,3), np.uint8);
#        for i in range(400):
#            if(i>199):
#                frame[i  , i,:] = [0,0,255]
#                frame[i  , 400-i-1,:] = [0,0,255]
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
                    if self.frame_idx % self.detect_interval == 0:
                        vis = frame.copy()
                        cv2.circle(vis, (x, y), 1, self.track_circle_color, -1)
#                    print("(x, y)", (x, y))
                    tr.append((x, y))
                    len_tr = self.get_len_track(tr)
                    if len(tr) > self.track_len:
                        del tr[0]
                    if(len_tr > 10):
                        m, b,new_points  = run_ransac(np.array(tr), estimate, lambda x, y: is_inlier(x, y, 0.1), 2, 3, len(tr), 20)
                        
                        if(m is None):
                            continue
                        a,b,c = m
                        corig = -a * new_points[0][0] - b * new_points[0][1]
                        c = get_diamond_c_from_original_coords(new_points[0][0]-22,new_points[0][1]-22,a,b,self.prms.w,self.prms.h)
                        coeffs = [a,b,c,1]#[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
                        new_line_coeffs.append(coeffs)
                        
                        xs = [0, width]
                        ys= [gety(xs[0], a,b,corig),gety(xs[1], a,b,corig)]
#                        print(ys)
#                        try:
#                            cv2.line( vis, (xs[0], int(ys[0])), (xs[1], int(ys[1])), (0,255,255), 2, 8 );        
#                        except:pass
                    new_tracks.append(tr)
                        

                self.tracks = new_tracks
                self.track_lines = new_lines
                self.track_lines_coeffs = new_line_coeffs
            if self.frame_idx % self.detect_interval == 0:
                sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=7)
                sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=7)
                magnitude = cv2.magnitude(sobelx, sobely) # computes sqrt(xi^2 + yi^2)
                phase = cv2.phase(sobelx,sobely,angleInDegrees=True) # computes angel between x and y
                
                H = get_orientation_matrix_way(magnitude, phase,1.0,8)
                
                if(len(self.B) == 0):
                    self.B = H
                else:
                    self.B = self.alpha * self.B + (1-self.alpha) * H
                
                H = background_test(self.B, H)
                H = np.sum(H,2)
                H = (H/np.max(H) * 255).astype('uint8')
                moving_lines= cv2.HoughLinesP(H,3,np.pi/180, 20, minLineLength=5,maxLineGap=1) 
                cv2.imshow('edges', H)
                if(moving_lines is not None):
                    for line in moving_lines:
                        x1,y1,x2,y2 = line[0]
                        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    new_moving_line_coeffs = []
                    for i in range(len(moving_lines)):
                        x1,y1,x2,y2 = moving_lines[i][0]
                        degree = 90
                        if(len(self.vp_1) > 0):
                            lt_vp1 = [self.vp_1[0] - x1, self.vp_1[1] - y1]
                            lt_cur_line = [x2-x1,y2-y1]
                            rad = np.arccos(np.dot(lt_cur_line, lt_vp1)/ ((np.sqrt(np.dot(lt_cur_line,lt_cur_line))) * np.sqrt(np.dot(lt_vp1, lt_vp1))))
                            degree = rad * 180 / np.pi
#                            print('between angle is', degree)
                        degree_threshold = 60
                        if(degree > degree_threshold and degree < 180-degree_threshold):
                            m, b,new_points  = run_ransac(np.array([[x1,y1],[x2,y2]]), estimate, lambda x, y: is_inlier(x, y, 0.1), 2, 3, 2, 20)
                        
                            if(m is None):
                                continue
                            a,b,c = m
                            corig = -a * new_points[0][0] - b * new_points[0][1]
                            c = get_diamond_c_from_original_coords(new_points[0][0]-22,new_points[0][1]-22,a,b,self.prms.w,self.prms.h)
                            coeffs = [a,b,c,1]#[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
                            new_moving_line_coeffs.append(coeffs)
                            
                            xs = [-width, width]
                            ys= [gety(xs[0], a,b,corig),gety(xs[1], a,b,corig)]
        #                        print(ys)
                            try:
                                cv2.line( vis, (xs[0], int(ys[0])), (xs[1], int(ys[1])), (0,255,255), 2, 8 );        
                            except:pass
                    self.moving_line_coeffs = new_moving_line_coeffs
                if(len(self.moving_line_coeffs) > 0):
                    result_moving_edges = diamond_vanish_with_lines(np.array(self.moving_line_coeffs), self.prms.w,self.prms.h,0.4, 321,1,self.space_old_moving_edges)
                    self.space_old_moving_edges = result_moving_edges["Space"]
                    resvps_moving_edges = np.int32(abs(result_moving_edges["CC_VanP"]))
                    self.vp_2  = resvps_moving_edges[0] - 1
                    print("detected vp =====", self.vp_2)
                if(len(self.track_lines_coeffs) > 0):
                    result = diamond_vanish_with_lines(np.array(self.track_lines_coeffs), self.prms.w,self.prms.h,0.4, 321,1,self.space_old)
                    self.space_old = result["Space"]
                    resvps = np.int32(abs(result["CC_VanP"]))
                    self.vp_1  = resvps[0] - 1
#                    print("detected vp =====", resvps)
#                    print(np.max(self.space_old))
                    for i in range(len(resvps)):
                        if(resvps[i][0] > 0 and resvps[i][1] > 0):
                            cv2.circle(vis, (resvps[i][0]-1, resvps[i][1]-1), 5, self.track_circle_color, -1)
                    
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 2, 0, -1)
            
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

                for i in range(len(self.tracks)):
                    cv2.polylines(vis, np.int32([self.tracks[i]]), False, self.track_line_color)
                if(len(self.vp_1) > 0 and len(self.vp_2) > 0):
                    for i in range(10):
                        random.seed(i)
                        line1, line2 = self.take_lane_towards_horizon((randrange(width), randrange(height)), length=30)
                        cv2.arrowedLine(vis, line1[1], line1[0]  , (0,0,255), 2, 8)
                        cv2.arrowedLine(vis, line2[1],  line2[0], (255,0,0), 2, 8)
            
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
#            cv2.imshow("edges", np.float32(edges * 255))
            
            ch = cv2.waitKey(1)
            if ch == 27:
                break
        

def main():
    videos_root = '/media/ixtiyor/New Volume/datasets/auto_callibration/d1'
    videos = os.listdir(videos_root)
    try:
        video_src =videos_root + "/" + videos[0] # "GOPR2036_half.mp4"
    except:
        video_src = 0

    App(video_src).run()
    cv2.destroyAllWindows()
    print('Done')
    
if __name__ == "__main__":
    main()

 

"""
        import gc


#    cv2.destroyAllWindows()
    img = cv2.imread("/media/ixtiyor/New Volume/datasets/bdd/bdd100k_images/bdd100k/images/10k/test/af962335-00000000.jpg",-1)
#    img = np.zeros((400, 400,3), np.uint8);
#    for i in range(400):
#        if(i<159):
#            img[i  , i,:] = [255,255,255]
#            img[i  , 400-i-1,:] = [255,255,255]
#            img[400-i-1  ,i ,:] = [255,255,255]
#    img = cv2.resize(img, (512,512 ))
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    space_size=  321
#    old_space = []
    w = img.shape[1]
    h = img.shape[0]
    normalization = 0.4
    m_norm = max([w, h])
    imgdraw = img.copy()
    lines = lsd.lsd(np.array(frame_gray, np.float32))
    lines = lines[:,0:4]
#        lines = np.array([[0,0, 50,50], [399,0,50,50]])
    line_coeffs = []
    max_iterations = 1
    for j in range(len(lines)):
        goal_inliers = len(lines[j])
        input_points = []
        input_points.append([lines[j][0],lines[j][1]])
        input_points.append([lines[j][2],lines[j][3]])
        input_points = np.array(input_points)
        m, b,new_points  = run_ransac(input_points+22, estimate, lambda x, y: is_inlier(x, y, 0.1), goal_inliers, max_iterations, 20)
        
        if(m is None):
            continue
        a,b,c = m
#            b, a,_,_ = fit_ellipse(input_points)
#            new_points = input_points
#            c = -a * new_points[0][0]  - b*new_points[0][1]
        
        if(len(np.shape(new_points))>1):
            c = get_diamond_c_from_original_coords(new_points[0][0],new_points[0][1],a,b,w,h)
#                c = -b * new_points[0][1] - a * new_points[0][0]
        else:
            c = get_diamond_c_from_original_coords(new_points[0],new_points[1],a,b,w,h)
#                c = -b * new_points[1] - a * new_points[0]
        coeffs = [a,b,c,1]#[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
        line_coeffs.append(coeffs)
    line_coeffs = np.array(line_coeffs)
    result = diamond_vanish_with_lines(line_coeffs,w,h,normalization, space_size,3,[])
#        result, line_coeffs, _ = diamond_vanish(img, normalization, space_size, 3, [])
    for j in range(lines.shape[0]):
        pt1 = (int(lines[j, 0]), int(lines[j, 1]))
        pt2 = (int(lines[j, 2]), int(lines[j, 3]))
        cv2.line(imgdraw, pt1,pt2, (0, 255, 255), 1)
    
    for j in range(len(line_coeffs)):
        a,b,c,w = line_coeffs[j]
        xs = [-1,1]
        ys= [gety(xs[0], a,b,c),gety(xs[1], a,b,c)]
#            if(not (np.any(np.isnan(xs)) or np.any(np.isnan(ys)))):
#                x1 = int((xs[0]/normalization * (m_norm-1) + w + 1) / 2)
#                y1 = int((ys[0]/normalization * (m_norm-1) + h + 1) / 2)
#                x2 = int((xs[1]/normalization * (m_norm-1) + w + 1) / 2)
#                y2 = int((ys[1]/normalization * (m_norm-1) + h + 1) / 2)
#                cv2.line( imgdraw, (x1,y1), (x2, y2), (255,255,0), 2, 8 );
#            ys= [gety(xs[0], a,b,c),gety(xs[1], a,b,c)]
#            cv2.line( imgdraw, (xs[0], int(ys[0])-1), (xs[1], int(ys[1])-1), (0,255,255), 2, 8 );        
    resvps = np.int32(abs(result["CC_VanP"]))-1
#        print("detected vp =====", resvps)
#    resvps = resvps[resvps[:,0] >0]
    for j in range(len(resvps)):
        cv2.circle(imgdraw, (resvps[j][0], resvps[j][1]), 5, [255,0,255], -1)
    #    except:
    ##        print('error')
    ##    dimimg = result['Space'] / np.max(result['Space']) * 255
#        old_space = result['Space']
#        del result
    gc.collect()
    cv2.imshow("img", imgdraw)
    ch = cv2.waitKey(0)
#    if(ch == 27): break
    cv2.destroyAllWindows()


# save line_coeffs to mat file
url ='/home/ixtiyor/Downloads/2013-BMVC-Dubska-source (2)/'

import numpy, scipy.io
scipy.io.savemat(url+'line_cooeffs.mat', mdict={'arr': line_coeffs})

#image = np.zeros((400, 400,3), np.float32);
#for i in range(400):
#    if(i>199):
#        image[i  , i,0] = 1.0
#        image[i  , 400-i-1,0] = 1.0
#        
#cv2.imshow('lk_track', image)
#ch = cv2.waitKey(0)
#cv2.destroyAllWindows()



#
#lines  = np.array_split(res, len_lines, axis=0)
#lines = np.array(lines)
#
#cv2.imshow('lk_track', edgest)
#ch = cv2.waitKey(0)
#cv2.destroyAllWindows()
"""