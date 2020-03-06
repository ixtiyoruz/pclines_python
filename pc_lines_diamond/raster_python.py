# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def convert_line_to_pc_line_endpoint(line):
    """
    line is 4 value array where 4 values are:
        a, b ,c w values
        from a x + b y + c = 0
        w is 1
    """
    a,b,c,w = line
    alpha = np.sign(a * b)
    beta = np.sign(b * c)
    thigma = np.sign(a * c)
    
    line_endp_ss_pp = [[alpha * a / (c + thigma * a), -alpha * c / (c + thigma * a)]]
    line_endp_st_pp = [[b/ (c + beta * b), 0]]
    line_endp_ts_pp = [[0, b/ (a + beta * b)]]
    line_endp_tt_pp = [[-alpha * a / (c + thigma * a), alpha * c / (c + thigma * a)]]
    res = np.r_[line_endp_ss_pp,line_endp_st_pp,line_endp_ts_pp,line_endp_tt_pp]
    return res

def fitt_ellipse(points):
    good_ellipse = 0
    int

if __name__ == "__main__":
#    test line is x + 2 where full written form is
    fig, axs = plt.subplots(1, figsize=(10,10))
    a, b,c = 300,-12, 5
    a1, b1, c1 = 2, -1, 8
    pxs = [-511, 511]
    pys = [pxs[0]*a + c, pxs[1] * a + c]
    pys1 = [pxs[0]*a1 + c1, pxs[1] * a1 + c1]
    axs.plot(pxs, pys,'k-')
    axs.plot(pxs, pys1,'k-')
    plt.show()
    
#    fig, axs = plt.subplots(1, figsize=(10,10))
    space_size = 321
    diamond_space = np.zeros((space_size,space_size))
    
    res = convert_line_to_pc_line_endpoint([a,b,c,1]) + space_size
#    axs.plot(res[:,0], res[:,1],'k-')
    res1 = convert_line_to_pc_line_endpoint([a1,b1,c1,1])+ space_size
#    axs.plot(res1[:,0], res1[:,1],'k-')
    diamond_space[np.int32(res[:,0]), np.int32(res[:,1])] = diamond_space[np.int32(res[:,0]), np.int32(res[:,1])] + 1
    diamond_space[np.int32(res1[:,0]), np.int32(res1[:,1])] = diamond_space[np.int32(res1[:,0]), np.int32(res1[:,1])] + 1
    
#    axs[0].plot(res[1:3,0], res[1:3,1],'k-')
#    axs[0].plot(res[2:4,0], res[2:4,1],'k-')
#    axs[0].plot([res[0,0], res[-1,0]],[res[0,1], res[-1,1]] ,'k-')
#    plt.show()
    
    
    

points = np.array([( 0 , 3),
    ( 1 , 2),
    ( 1 , 7),
    ( 2 , 2),
    ( 2 , 4),
    ( 2 , 5),
    ( 2 , 6),
    ( 2 ,14),
    ( 3 , 4),
    ( 4 , 4),
    ( 5 , 5),
    ( 5 ,14),
    ( 6 , 4),
    ( 7 , 3),
    ( 7 , 7),
    ( 8 ,10),
    ( 9 , 1),
    ( 9 , 8),
    ( 9 , 9),
    (10,  1),
    (10,  2),
    (10 ,12),
    (11 , 0),
    (11 , 7),
    (12 , 7),
    (12 ,11),
    (12 ,12),
    (13 , 6),
    (13 , 8),
    (13 ,12),
    (14 , 4),
    (14 , 5),
    (14 ,10),
    (14 ,13)])
    
    
    
    
plt.plot(points[:,0], points[:,1], '.')
plt.show()

x = points[:,0][:,np.newaxis]
y = points[:,1][:,np.newaxis]
D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
S = np.dot(D.T,D)
C = np.zeros([6,6])
C[0,2] = C[2,0] = 2; C[1,1] = -1
E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
n = np.argmax(np.abs(E))
a = V[:,n]
