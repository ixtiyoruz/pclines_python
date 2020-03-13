# -*- coding: utf-8 -*-
#import nu
def get_diamond_c_from_original_coords(x,y, a,b, width, height,padding =22, radius=22):
    """
    formula is 
    ax + by + c = 0
    x = (xorig + padding - wc) / norm
    y = (yorig + padding - hc) / norm
    where x, y is in diamond space
    """
    wc = (width + padding * 2 - 1) / 2
    hc = (height + padding * 2 - 1) / 2
    norm = max(wc, hc) - radius
    
    c = -(a * (x+padding + padding -wc)) / norm - (b * (y +padding - hc)) / norm
    return c

def get_original_y_from_diamond_space(x, a, b, c, width, height, radius=22):
    """
    formula is 
    ax + by + c = 0
    x = (xorig + padding - wc) / norm
    y = (yorig + padding - hc) / norm
    where x, y is in diamond space
    """
    wc = (width +44-1)/2
    hc = (height+44-1)/2
    norm = max(width, height) - radius
    y = - a/b * (x + 22 - wc) - c/b * norm - 22 + hc
    return y
def get_original_c_from_original_points(x,y, a,b):
#    print('inputted points are ', np.shape(o))
#    xt = x - np.average(x)
#    yt = y - np.average(y)
#    print(xt)
#    D =  np.c_[xt, yt]
    c = -b * y - a *x
    return c

def gety(x,a,b,c):
    """
    ax  + by + c = 0
    """
    y = (-a*x - c) / b
    return y
def get_coeffs(points):
    goal_inliers = len(points)
    max_iterations = 3
    m, b,new_points  = run_ransac(points, estimate, lambda x, y: is_inlier(x, y, 0.1), goal_inliers, max_iterations, 20)
#    print(m)
    a,b,c = m
    c = -b * new_points[0][1] - a * new_points[0][0]
    return a,b,c
