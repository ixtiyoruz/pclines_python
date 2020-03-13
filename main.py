# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from rectification import *
import time
import math

#/media/ixtiyor/New Volume/datasets/auto_callibration/d1
videos_root = '/media/ixtiyor/New Volume/datasets/auto_callibration/d1'
videos = os.listdir(videos_root)

def apply_hough_transform():
    lines = []
    
    # to do 
    pass
def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)        
def calculate_length(p1, p2):
    return math.sqrt((p1[0]-p2[0]) * (p1[0]-p2[0]) + (p1[1]-p2[1]) * (p1[1]-p2[1]))

def main():
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.6,
                       minDistance = 7,
                       blockSize = 7 )
    
    color = [[0,0,255], [255,0,0], [0,0,0]]
    cap = cv2.VideoCapture(videos_root + "/" + videos[2])
#    cv2.openWindow()
    _, frame = cap.read()
    frame = cv2.resize(frame, (512,512 ))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    old_gray = cv2.GaussianBlur(old_gray,(3,3),0)
#    old_gray = cv2.Sobel(old_gray,cv2.CV_8U,1,1,ksize=5)  # x
#    p0 = detect_feature_ShiTomasi(old_gray, qualityLevel=10)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    frame_no = 0
    mask = np.zeros_like(frame)
#    p0 = np.array(p0, np.float32)
    diff_xs1 = []
    diff_xs2 = []
    while(True):
        
        r, frame  = cap.read()
        frame = cv2.resize(frame, (512,512 ))
#        start = time.time()
        if int(frame_no)%10 == 0:
#            p0 = detect_feature_ShiTomasi(old_gray, qualityLevel=10)
#            p0 = np.array(p0, np.float32)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#        print('number of det fet is' ,np.shape(p0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
#        gray = cv2.GaussianBlur(gray,(3,3),0)
#        gray = cv2.Sobel(gray,cv2.CV_8U,1,1,ksize=5)  # x
#        gray = cv2.Canny(gray,100,200)
#        cv2.imshow('gray', gray)
#        cv2.imshow('oldgray', old_gray)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0,None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            if(d>0.7*frame.shape[0] or b>0.7*frame.shape[0]):continue
            if(d<0.2*frame.shape[0] or b<0.2*frame.shape[0]):continue
            diff_xs1.append(calculate_length([a,b], [c,d]))
            if(d-b > 0 and abs(c-a) <= 20):
                mask = cv2.line(mask, (a,b),(c,d), color[0], 2)
            elif(d-b < 0 and abs(c-a) <= 20):
                mask = cv2.line(mask, (a,b),(c,d), color[1], 2)
#                diff_xs2.append(abs(c-a))
            else:
                mask = cv2.line(mask, (a,b),(c,d), color[2], 2)
                
            frame = cv2.circle(frame,(a,b),2,color[1],-1)
        print(np.mean(diff_xs1))
        frame = cv2.add(frame,mask)    
#        edgelets1 = compute_edgelets(frame)
        
#        print('time usage: ', time.time() - start)
#        draw_edgelets(frame, edgelets1) # Visualize the edgelets
        old_gray = gray.copy()
        p0 = good_new.reshape(-1,1,2)     #np.append(p0, good_new.reshape(-1,1,2), 0)#good_new.reshape(-1,1,2)     
        frame_no = frame_no + 1
        
        cv2.imshow('edges', frame)
        k = cv2.waitKey(3)
        if(k == 27):
            break
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
