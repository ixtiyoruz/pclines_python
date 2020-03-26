# -*- coding: utf-8 -*-

import darknet_video
import cv2
import os 
#from pylsd import lsd
import numpy as np
import pickle 
from pc_lines_diamond.utils import take_lane_towards_horizon, calculate_distance,rotationMatrixToEulerAngles
from random import randrange
import random
import math

videos_root = '/media/ixtiyor/New Volume/datasets/auto_callibration/videos/g1'
videos = os.listdir(videos_root)
try:
    video_src =  videos_root + "/" + videos[1]  # "GOPR2036_half.mp4" #
except: 
    video_src = 0
cap = cv2.VideoCapture(video_src)
print('creating sort object')
#print('starting the detection')


with open('calibration.pickle','rb') as f:
    loaded_obj = pickle.load(f)
vp_1 = loaded_obj['vp1']
vp_2 = loaded_obj['vp2']
vp_3 = loaded_obj['vp3']
f = loaded_obj['f']
R = loaded_obj['R']


while(True):
    _, frame = cap.read()
    if(_):
#        frame = cv2.imread("dasdadasdasdasd.jpg",-1); #"/media/ixtiyor/New Volume/datasets/bdd/bdd100k_images/bdd100k/images/10k/test/af962335-00000000.jpg",-1)
        detections = darknet_video.detect_from_image(frame)
            
        original_height, original_width , _ = np.shape(frame)
        detections_new, output_formatted = darknet_video.resize_detections(detections, original_height, original_width)
        ## second one is x1,y1, x2,y2
        ##        print(detections)       
        #frame = darknet_video.cvDrawBoxes(detections_new,frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for detection in detections_new:
            x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
            xmin, ymin, xmax, ymax = darknet_video.convertBack(
                float(x), float(y), float(w), float(h))
        #    xmin = 150
        #    xmax = 205
        #    ymin=231 
        #    ymax=257
        #            padding = 3    
        #            frame_gray = frame_gray[ymin-padding:ymax+padding, xmin-padding:xmax+padding]
        #            print(np.shape(frame_gray))
        #            if(frame_gray.shape[0] * frame_gray.shape[1] > 0):
        #                lines = lsd.lsd(np.array(frame_gray, np.float32))
        #                lines = lines[:,0:4] 
        #                for j in range(lines.shape[0]):
        #                    if(int(lines[j, 0] + xmin) < original_height and int(lines[j, 1] + ymin) < original_width):
        #                        pt1 = (int(lines[j, 0] + xmin - padding), int(lines[j, 1] + ymin - padding))
        #                        pt2 = (int(lines[j, 2] + xmin - padding), int(lines[j, 3] + ymin - padding))
        #                        length = np.sqrt(np.sum(np.square(np.array(pt1) - np.array(pt2))))
        #                        print(length)
        #                        if(length > 10):
        #                            cv2.line(frame, pt1,pt2, (0, 255, 255), 1)
        #                cv2.imshow('frame', frame_gray)
        #            print(ymin, vp_1)
            angles = rotationMatrixToEulerAngles(R,False)            
            thigma = math.atan( vp_1[1] / math.sqrt(vp_1[0]* vp_1[0] + f*f))
            print('thigma = ', thigma)
            dist_ = calculate_distance(vp_1, ymax, f,3.2672, thigma)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            cv2.putText(frame,
                    detection[0].decode() + " " + str(np.round(dist_, 2)) + " m",
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 255, 255], 1)
        for i in range(3):
            random.seed(i)
            line1 = []
            line2 = []
            line3 = []
            point_r = (randrange(original_width), randrange(original_height))
            if(len(vp_1) > 0):
                line1 = take_lane_towards_horizon(point_r,vp_1, length=30)
            if(len(vp_2) > 0):
                line2 = take_lane_towards_horizon(point_r,vp_2, length=30)   
            if(len(vp_3) > 0):
                line3 = take_lane_towards_horizon(point_r,vp_3, length=30)   
                
            if(len(line1) > 0):
                cv2.arrowedLine(frame, line1[1], line1[0]  , (0,0,255), 2, 8)
            if(len(line2) > 0):
                cv2.arrowedLine(frame, line2[1],  line2[0], (255,0,0), 2, 8)
            if(len(line3) > 0):
                cv2.arrowedLine(frame, line3[1],  line3[0], (0,255,0), 2, 8)     
        cv2.imshow('img', frame)
        ch = cv2.waitKey(500)
        if(ch == 27):
            cv2.destroyAllWindows()
            break

