from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
old_detections = None
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):

    for detection in detections:
#        print("detection[2] = ", detection[2])
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]" ,
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def cvDrawBoxes_with_distance(detections, img):

    for detection in detections:
        x, y, w, h, distance = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3],\
            detection[4]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]" +  " " + str(round(distance, 2))+" m",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img
netMain = None
metaMain = None
altNames = None
darknet_image = None

def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./data/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    cap = cv2.VideoCapture(0)#"test.mp4")
    cap.set(3, 720)
    cap.set(4, 720)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        w_o, h_o,_ = np.shape(frame_rgb)        
        # print(w_o, h_o)
        # amount = 0.6
        # frame_rgb = frame_rgb[int((1-amount)*w_o):w_o, 0:h_o]
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        #print(metaMain)
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.45)
        detections = resize_detections(detections, w_o,h_o,darknet.network_width(netMain), darknet.network_height(netMain))
        image = cvDrawBoxes(detections, frame_rgb)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        if(cv2.waitKey(3) > 0):break
    cap.release()
    # out.release()

init= None   
def resize_detections(detections, w_o, h_o, w_c=416, h_c=416):
    aspect_r_w = w_o*1.0/w_c
    aspect_r_h = h_o*1.0/h_c
    #print(aspect_r_w, aspect_r_h)
    new_detections = []
    new_detections_1 = [] # for the format x1, y1 ,x2, y2
    for detection in detections:
        x = detection[2][0] * aspect_r_h
        y= detection[2][1] * aspect_r_w
        w = detection[2][2] * aspect_r_h
        h = detection[2][3 ] * aspect_r_w
        x1 = x - w//2
        y1 = y - h//2
        x2 = x + w//2
        y2 = y + h//2
        
                # print(str(detection[0]))
        # if(detection[0].decode() == 'cup'):
        new_detections_1.append((x1, y1, x2, y2))
        new_detections.append((detection[0], detection[1], (x, y, w, h)))
    return new_detections, new_detections_1

def detect_from_image(image_frame, thresh=0.5):
    global metaMain, netMain, altNames, init, darknet_image
    if(init is None):
        print('reloading darknet module')
        configPath = "./cfg/yolov3.cfg"
        weightPath = "./yolov3.weights"
        metaPath = "./data/coco.data"
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(metaPath)+"`")
        if netMain is None:
            netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        if darknet_image is None:
            # Create an image we reuse for each detect
            darknet_image = darknet.make_image(darknet.network_width(netMain),darknet.network_height(netMain),3)
    frame_resized = cv2.resize(image_frame,
                                (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)
    
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    #print(metaMain)
    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=thresh)
    init = True
    return detections
if __name__ == "__main__":
    YOLO()
