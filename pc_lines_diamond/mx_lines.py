# -*- coding: utf-8 -*-


# you can use 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pc_lines_diamond.ransac.ransac import *

def augment(xys):
    axy = np.ones((len(xys), 3))
    axy[:, :2] = xys
    return axy

def estimate(xys):
    axy = augment(xys[:2])
    return np.linalg.svd(axy)[-1][-1, :]

def is_inlier(coeffs, xy, threshold):
    return np.abs(coeffs.dot(augment([xy]).T)) < threshold
#def get_lines_coeffs(lines_xy_coordinates, radius = 22, img_shape=[512,512]):
#    """
#    linexy_coordinates can be n x ? array n is number of lines
#    ? is number of points in that line may be any numbers
#    """
#    rads_size =1
#    patch_size= radius * 2 + 1
#    height, width = img_shape
#    
#    dist = np.zeros([len(lines_xy_coordinates, patch_size, patch_size)])
#    # mask with distances
#    for i in range(patch_size):
#        for j in range(patch_size):
#            dist[i,j] = max(abs(i - radius), abs(j-radius))
#    
#    w_c = (width - 1) / 2.0
#    h_c = (height - 1)/ 2.0
#    
#    norm = max(w_c, h_c) - radius
#    
##    for i in np.arange(radius, width - radius, 1):
##        for j in np.arange(radius, height - radius,1):
#    for i in range(len(lines_xy_coordinates)):
#        results  = lines_xy_coordinates[i]
#        
#        # fit ellipse
        

def fit_ellipse(results, distances):
    good_ellipse = True
    vec_data = np.zeros((4))
#    dist = 1 
#    while(good_ellipse):
    cov_data = np.zeros((3))
    x_mean = 0
    y_mean = 0
    
#        size = 0
    for i in range(len(results)):
        x_mean  = x_mean + results[i][0]
        y_mean = y_mean + results[i][1]
    x_mean = x_mean/len(results)
    y_mean = y_mean/len(results)
    xx = 0
    yy= 0
    xy = 0
    
    for i in range(len(results)):
        x = results[i][0]- x_mean
        y = results[i][1] - y_mean
        xx =xx + x* x
        yy = yy + y * y
        xy = xy + x * y

    cov_data[0] = xx/ len(results)
    cov_data[1] = yy/ len(results)
    cov_data[2] = xy/ len(results)
    cov = np.array([[cov_data[0], cov_data[2]],
            [cov_data[2], cov_data[1]]])
    _,vec_data,_ =  np.linalg.svd(cov)
#        print(good_ellipse)
    return vec_data

def eigens(cov_data):
#    ellipse_threshold = 1
    vec_data = np.zeros((4))
#    trace = cov_data[0] + cov_data[2]
#    det = cov_data[0] * cov_data[2]  - cov_data[1] - cov_data[1]
#    s = np.sqrt(trace * trace /4 -det)
#    vals_data = np.array([trace/2 -s, trace/2 + s])
    
    if(abs(cov_data[1]) > 0.001):
        vec_data[0] = vec_data[0] - cov_data[2]
        vec_data[1] = cov_data[1]
        
        n = np.sqrt(vec_data[0] * vec_data[0] + vec_data[1]*vec_data[1])
        vec_data[0] = vec_data[0]/ n
        vec_data[1] = vec_data[1] / n
    else:
        vec_data[0] = int(cov_data[0] < cov_data[1])
        vec_data[1] = vec_data[0] - 1
    vec_data[2] = vec_data[1]
    vec_data[3] = -vec_data[0]
    
#    print(vals_data, vals_data[1]/vals_data[0] < ellipse_threshold)
    return vec_data


   
#
#    
#points = np.array([( 0 , 3),
#    ( 1 , 2),
#    ( 1 , 7),
#    ( 2 , 2),
#    ( 2 , 4),
#    ( 2 , 5),
#    ( 2 , 6),
#    ( 2 ,14),
#    ( 3 , 4),
#    ( 4 , 4),
#    ( 5 , 5),
#    ( 5 ,14),
#    ( 6 , 4),
#    ( 7 , 3),
#    ( 7 , 7),
#    ( 8 ,10),
#    ( 9 , 1),
#    ( 9 , 8),
#    ( 9 , 9),
#    (10,  1),
#    (10,  2),
#    (10 ,12),
#    (11 ,8),
#    (11 , 7),
#    (12 , 7),
#    (12 ,11),
#    (12 ,12),
#    (13 , 6),
#    (13 , 8),
#    (13 ,12),
#    (14 , 4),
#    (14 , 5),
#    (14 ,10),
#    (14 ,13)])
#        
    
def get_coeffs(points):
#    print('inputted points are ', np.shape(o))
#    xt = x - np.average(x)
#    yt = y - np.average(y)
#    print(xt)
#    D =  np.c_[xt, yt]
    goal_inliers = len(points)
    max_iterations = 3
    m, b  = run_ransac(points, estimate, lambda x, y: is_inlier(x, y, 0.1), goal_inliers, max_iterations, 20)
#    print(m)
    a,b,c = m
    c = -b * np.average(points[:,1]) - a *np.average(points[:,0])
    return a,b,c


#ax + by + c=0
#y = -1(ax+c)/ b
def gety(x, a,b,c):
    return -(a*x + c)/b

if __name__ == '__main__':
    
   
    points =  np.array([[270.89157, 303.38962],
       [272.86075, 312.51575],
       [275.07196, 322.43488],
       [277.3726 , 333.2462 ],
       [280.0113 , 344.9997 ],
       [282.73657, 357.89746]])
    #plt.show()

    #n = np.argmax(np.abs(E))
    #a = V[:,n]
    
#    U, S, V = np.linalg.svd()
    a,b,c =get_coeffs(points)
    
    
    
    #a, b = fit_ellipse(points,[])
    #points_hom = np.c_[points[:, 0],points[:, 1], np.ones(len(points))]
    #u,s, v = 
    width = 512
    height = 512
    norm = max(width, height)
    h_c = (height - 1)/ 2
    w_c = (width - 1) / 2
#    j = np.mean(points[:, 1])
#    i = np.mean(points[:, 0])
#    c = -b * np.average(y) - a *np.average(x)
    xs = [0,311]
    
#    ransac = linear_model.RANSACRegressor()
    ys= [gety(xs[0], a, b, c),gety(xs[1], a, b, c)]
    fig, axs = plt.subplots(1, figsize=(10,10))
    plt.plot(points[:,0], points[:,1], '.')
    #plt.plot([0, height-1],[0, width-1], '.')
    axs.plot(np.array(xs), np.array(ys),'k-')
    plt.show()