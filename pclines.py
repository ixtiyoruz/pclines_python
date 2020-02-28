import math 
import functools
import numpy as np
from pylsd import lsd
import cv2 as cv
from mex_files.alignments_slow_python import use_alignments_slow
from gmm_mml import GmmMml
import matplotlib.pyplot as plt
import matplotlib

# there is a problem with sign function in python so this is a workaround
# https://stackoverflow.com/questions/1986152/why-doesnt-python-have-a-sign-function
sign = functools.partial(math.copysign, 1)

class params:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.LENGTH_THRESHOLD = 1.01 # param to change
        self.LENGTH_THRESHOLD = math.sqrt(self.w + self.h)/self.LENGTH_THRESHOLD 
        self.GMM_KS = [4]
        
def detect_vps(frame_gray, prms, frame_to_draw=None):
    lines = lsd.lsd(np.array(frame_gray, np.float32))
    lines = lines[:,0:4]
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        if(not frame_to_draw is None):
            cv.line(frame_to_draw, pt1,pt2, (0, 0, 255), 1)
    denoised_lanes = denoise_lanes(lines, prms)
    if(len(denoised_lanes) > 0):
        points_staright, points_twisted = convert_to_PClines(denoised_lanes, prms)
    else:
        points_staright , points_twisted = [], []
    points_staright = points_staright[0:100]
    print('inputshape',np.shape(points_staright), np.shape(points_staright.transpose().ravel()))
    detections_straight, m1, b1 =  find_detections(points_staright, prms)
    detections_twisted, m2, b2 =  find_detections(points_twisted, prms)
    
    # gather initial vanishing point detections
def get_ellipse_endpoints(mu, cov, level=2, draw=False):
        uu, ei, vv = np.linalg.svd(cov)
        a = np.sqrt(ei[0] * level * level)
        b = np.sqrt(ei[1] * level * level)
        theta = np.array([0, np.pi])
        xx = a * np.cos(theta)
        yy = b * np.sin(theta)
        cord = np.c_[xx.T, yy.T].T
    #     cord = uu * cord
        x0 = cord[0][0]  + mu[0]
        x1 = cord[0][1] + mu[0]
        y0 = cord[1][0] + mu[1]
        y1 = cord[1][1]  + mu[1]
        thetas = np.arange(0, 2*np.pi, 0.01)
        xxs = a * np.cos(thetas)
        yys = b * np.sin(thetas)
        cords = np.c_[xxs.T, yys.T].T 
        if(draw):
            plt.plot(cords[0] +mu[0] , cords[1] +mu[1])
        return np.array([x0, y0, x1, y1])
        
def run_mixtures(points, Ks=[20, 40, 60], filename="candidate_pairs.txt", draw=False):
    # Runs Figueiredo et al. GMM algorithm with different parameters (number of
    # Gaussians). The endpoints of the ellipses found are saved as candidates
    # for the alignment detector.
    # Parameters:
    # - points: list of 2D points
    # - Ks: number of Gaussians to try (example: [20 40 60])
    # - file_path: path where to save a text file with the obtained pairs of
    # points
#    print('run_mixtures started\n')
    points = np.round(points) 
#    print(np.shape(points))
    points = np.vstack({tuple(row) for row in points}) # only getting unique rows
#    print(np.shape(y))
#    print(np.any(np.isnan(points)))
#    npoints = len(y[0])
    
    all_bestpairs = []
    
    for k in range(len(Ks)):
        K = Ks[k]
        unsupervised=GmmMml(max(2,K-7),K,0,1e-4, 0,  max_iters=2)        
        new_labels = unsupervised.fit_transform(points)
        new_labels = np.argmax(new_labels,-1)
        if(draw):
            fig, ax = plt.subplots()
#            print(np.shape(points[:,0]), np.shape(points[:,1]), np.shape(new_labels))
            plt.scatter(points[:,0],points[:,1], c=new_labels, alpha=0.3,s=10)
        
        best_pairs = np.zeros((unsupervised.bestk, 4))
        for comp in range(unsupervised.bestk):
            best_pair = get_ellipse_endpoints(unsupervised.bestmu[comp],unsupervised.bestcov[:,:, comp], 2, draw=False)
            best_pairs[comp, :] = best_pair
        all_bestpairs.append(best_pairs)
        if(draw):
            plt.show()
    return all_bestpairs

def find_detections(points, prms):
    # now skip to slow version
    M = np.max(points)
    m = np.min(points)
    
    points = (points - m)/(M - m) * 512 # this version of the alignment detector expects a 512 x 512 domain
    N = len(points)
    candidates = run_mixtures(points, prms.GMM_KS,'')
    print(candidates)
    
    points = list(points.transpose().ravel())
    print(np.shape(candidates))
    return [], [], []
    # now the slow one
    detections,n_out = use_alignments_slow(points, 2, N)
    
    detections = np.array(np.array_split(detections, n_out))
    if(not len(detections) == 0):
        dets = detections[:, 0:4]
        dets = dets/512 * (M -m)+ m
        detections[:, 0:4] = dets
        x1= dets[:,0]
        y1= dets[:,1]
        x2= dets[:,2]
        y2= dets[:,3]
        
        dy = y2 - y1
        dx = x2 - x1
        m = dy/dx
        b = (y1 * x2 - y2 * x1)/dx
        return dets, m, b 
    else:
        return [], [], []
    
#    return detections
def denoise_lanes(lines, prms):
    """
    lanes: array with shape n x 4
    [x1, y1 , x2, y2]
    """
    new_lines = np.array(lines)
    
    if(len(new_lines) > 0):
        lengths =np.sum(np.sqrt([np.power(new_lines[:, 2] - new_lines[:,0], 2),np.power(new_lines[:,3] - new_lines[:,1], 2)]), 0)
    else:
        return []
    #print(lengths)

    # now denoise according to length_threshold 
    lengths  = np.ravel(lengths)
    matched_args = np.where(lengths > prms.LENGTH_THRESHOLD)
    # un_matched_args = np.where(lengths <= prms.LENGTH_THRESHOLD)
    lines_large = new_lines[matched_args]
    # print(np.shape(lines_large))
    # lines_short = new_lines[un_matched_args]
    return lines_large

def convert_to_PClines(lines, prms):
    """
        lines in the shape of n x 4 or n x 2
        where 
        4 values indicates:
        x, y, x1, y1 : defining coordinates of the line
    """
    H = prms.h 
    W = prms.w 
    L = len(lines)
    points_straight = PCLines_straight_all(lines/ np.tile([W, H, W, H],[L, 1]))
    points_twisted = PCLines_twisted_all(lines/ np.tile([W, H, W, H],[L, 1]))
    args_4_del_strait = np.where((points_straight[:,0]>2) | (points_straight[:,1]>2)\
                               | (points_straight[:,0]<-1) | (points_straight[:,1]<-1) \
                               | (np.isnan(points_straight[:,0])) | (np.isnan(points_straight[:,1]) ))[0]

    args_4_del_twisted = np.where((points_twisted[:,0]>1) | (points_twisted[:,1]>1)| \
                                   (points_twisted[:,0]<-2) | (points_twisted[:,1]<-2) | \
                                   (np.isnan(points_twisted[:,0])) | (np.isnan(points_twisted[:,1])))[0]
    
    if(len(args_4_del_strait) > 0):
        points_straight = np.delete(points_straight, args_4_del_strait, axis=0) 
#        print(np.shape(points_straight))
    if(len(args_4_del_twisted) > 0):
        points_twisted = np.delete(points_twisted, args_4_del_twisted, axis=0)         
    return [points_straight, points_twisted]

def PCLines_straight_all(l):
    """
        transforms line as [x1,y1, x2, y2] or a point as [x,y] with PCLines straight
        transform coordinates should be normalized
    """ 
    # print('entered data shape', np.shape(l))
    d = 1.0 # arbitrary distance between vertical axes x and y
    L = len(l[0])
    if(L == 4):
        x1 = l[:, 0]
        y1 = l[:, 1]
        x2 = l[:, 2]
        y2 = l[:, 3]
        dy = y2 - y1
        dx = x2 - x1
        m = dy / dx
        b = (y1 * x2 - y2 * x1)/ dx 
        PCline = np.tile(d, [len(b),1])
        PCline = np.append(PCline, np.reshape(b, (len(b), 1)), 1)
        PCline = np.append(PCline, np.reshape(1-m, [len(m),1]), 1) # homogeneous coordinates
        
        u = PCline[:, 0] / PCline[:,2]
        v = PCline[:, 1] / PCline[:,2]
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res
    elif(L == 2):
        """it is a point"""
        x = l[:, 0]
        y = l[:, 1]
        b = x 
        m = (y - x) /d
        u = m 
        v = b 
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res

def PCLines_twisted_all(l):
    """
        transforms line as [x1,y1, x2, y2] or a point as [x,y] with PCLines twisted
        transform coordinates should be normalized
    """ 

    d = 1 # arbitrary distance between vertical axes x and y
    L = len(l[0])
    if(L == 4):
        x1 = l[:, 0]
        y1 = l[:, 1]
        x2 = l[:, 2]
        y2 = l[:, 3]
        dy = y2 - y1
        dx = x2 - x1
        m = dy / dx
        b = (y1 * x2 - y2 * x1)/ dx 
        PCline = np.tile(-d, [len(b),1])
        PCline = np.append(PCline, -1 *np.reshape(b, (len(b), 1)), 1)
        PCline = np.append(PCline, np.reshape(1+m, [len(m),1]), 1) # homogeneous coordinates
        u = PCline[:, 0] / PCline[:,2]
        v = PCline[:, 1] / PCline[:,2]
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res
    elif(L == 2):
        """it is a point"""
        x = l[:, 0]
        y = l[:, 1]
        b = x 
        m = (y + x) /d
        u = m 
        v = b 
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res
