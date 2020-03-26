#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:20:22 2020

@author: ixtiyor
"""
import cv2
import os
import numpy as np
from pc_lines_diamond.mx_lines import  fit_ellipse, gety,get_coeffs
from pc_lines_diamond.ransac.ransac import run_ransac, estimate, is_inlier
import math
import time
#def length(v):
#  return math.sqrt(dotproduct(v, v))
def auto_canny(image, sigma=0.43):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
def get_orientation(mag, ori,thresh=1.0, num_bins=8):
    h, w = np.shape(mag)
    orientation_map = np.zeros((*np.shape(mag), num_bins))
    for i in range(h):
        for j in range(w):
            mag_pixel = mag[i,j]
            if(mag_pixel > thresh):
                 oriPixel = ori[i,j]
                 bins = np.arange(0,361,360/(num_bins))
                 for ib in range(len(bins)-1):
                     if(ib ==1 or len(bins)-1 - 1):continue
#                     print('checking from ', bins[ib] , ' to ', bins[ib+1] )
                     if(oriPixel >= bins[ib] and oriPixel < bins[ib+1]):
                         
                         orientation_map[i,j,ib] = mag_pixel
                         break
#            print(i, j)
    return orientation_map
def get_orientation_matrix_way(mag, ori,thresh=1.0, num_bins=8):
    h, w = np.shape(mag)
    orientation_map = np.zeros((*np.shape(mag), num_bins))
    bins = np.arange(0,361,360/(num_bins))
    for ib in range(len(bins)-1):
        rws,cols = np.where((ori>bins[ib]) & (ori<bins[ib+1]))
        orientation_map[rws,cols, ib] = mag[rws,cols]
    return orientation_map
def background_test(B, H, t1=11065969, t2=12065969):
    diff = abs(H - B)
#    img = np.zeros((*np.shape(H), 3))
#    min_val = np.min(H)
#    print(np.max(diff), np.min(diff))
    rws_t2,cols_t2,bins_t2 = np.where(diff < t2)
    rws_t1,cols_t1,bins_t1 = np.where(H<t1)
#    rws_t3,cols_t3,bins_t3 = np.where()
    H[rws_t1, cols_t1, bins_t1] = 0
#    H[rws_t3, cols_t3, bins_t3] = 0
    H[rws_t2, cols_t2, bins_t2] = 0
#    imgp[rws_t2, cols_t2] = []
    return H
if(__name__ == '__main__'):
#    videos_root = '/media/ixtiyor/New Volume/datasets/auto_callibration/d1'
#    videos = os.listdir(videos_root)
    try:
        video_src = "GOPR2036_half.mp4"#videos_root + "/" + videos[1] # "GOPR2036_half.mp4"
    except:
        video_src = 0
        
    cam = cv2.VideoCapture(video_src)
    _ret, frame = cam.read()
#    frame = cv2.resize(frame, (512,512 ))
    width = frame.shape[1]
    height = frame.shape[0]
    #i = 0
    B = []
    alpha = 0.95
    while True:
        _ret, frame = cam.read()
#        frame = cv2.resize(frame, (512,512 ))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start = time.time()
        frame_gray = auto_canny(frame_gray, 0.1)
    #    print(i)
        sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=11)
        sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=11)
        
    #    kernely = np.array([[1,1,1],
    #                        [0,0,0],
    #                        [-1,-1,-1]])
    #    
    #    kernelx = np.array([[1,0,-1],
    #                        [1,0,-1],
    #                        [1,0,-1]])
    #    
    #    sobelx = cv2.filter2D(frame_gray, cv2.CV_64F, kernelx)
    #    sobely = cv2.filter2D(frame_gray, cv2.CV_64F, kernely)
        
    #    cv2.imshow('Gradients_X',sobelx)
    #    cv2.imshow('Gradients_Y',sobely)
        
        magnitude = cv2.magnitude(sobelx, sobely) # computes sqrt(xi^2 + yi^2)
        phase = cv2.phase(sobelx,sobely,angleInDegrees=True) # computes angel between x and y
        
        H = get_orientation_matrix_way(magnitude, phase,1.0,8)
        
        if(len(B) == 0):
            B = H
        else:
            B = alpha * B + (1-alpha) * H
        
        H = background_test(B, H)
        H = np.sum(H,2)
        H = (H/np.max(H) * 255).astype('uint8')
        vp_1 = [width/2, 0]
        print('time spent', time.time() - start)
    #    edges = cv2.Canny(H.astype('uint8'),50,150,apertureSize = 3)
    #    np.as
    #     This returns an array of r and theta values 
#        lines = cv2.HoughLinesP(H,1,np.pi/180, 1, minLineLength=8,maxLineGap=3) 
#        
#        if(lines is not None):
#            # The below for loop runs till r and theta values  
#            # are in the range of the 2d array 
##            print(np.shape(lines))
#            for line in lines:
#                x1,y1,x2,y2 = line[0]
#                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
#            for i in range(len(lines)):
#                    
#                    x1,y1,x2,y2 = lines[i][0]
#                    lt_vp1 = [vp_1[0] - x1, vp_1[1] - y1]
#                    lt_cur_line = [x2-x1,y2-y1]
#                    rad = np.arccos(np.dot(lt_cur_line, lt_vp1)/ ((np.sqrt(np.dot(lt_cur_line,lt_cur_line))) * np.sqrt(np.dot(lt_vp1, lt_vp1))))
#                    degree = rad * 180 / np.pi
##                            print('between angle is', degree)
#                    degree_threshold = 45
#                    if(degree > degree_threshold and degree < 180-degree_threshold):
#                        m, b,new_points  = run_ransac(np.array([[x1,y1],[x2,y2]]), estimate, lambda x, y: is_inlier(x, y, 0.1), 2, 3, 2, 20)
#    #                    print(m, b)
#                        if(m is None):
#                            continue
#                        a,b,c = m
#                        corig = -a * new_points[0][0] - b * new_points[1][1]
#    #                    c = get_diamond_c_from_original_coords(new_points[0]-22,new_points[1]-22,a,b,self.prms.w,self.prms.h)
#    #                    coeffs = [a,b,c,1]#[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
#    #                    new_moving_line_coeffs.append(coeffs)
#                        
#                        xs = [-width, width]
#                        ys= [gety(xs[0], a,b,corig),gety(xs[1], a,b,corig)]
##                        print(ys)
#                        try:
#                            cv2.line( frame, (xs[0], int(ys[0])), (xs[1], int(ys[1])), (0,255,255), 2, 8 );        
#                        except:pass
        cv2.imshow('lk_track', frame)
        cv2.imshow('edges',H)
        
        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.destroyAllWindows()        
            break
    
