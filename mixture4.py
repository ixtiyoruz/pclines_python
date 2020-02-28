import numpy as np
import sys

def mixture4(y, kmin, kmax, regularize, th, covoption, npoints):
    #
    """
     Usage syntax:
     [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures3(y,kmin,kmax,regularize,th,covoption)
    
     Inputs:
    
     "y" is the data; for n observations in d dimensions, y must have
     d lines and n columns.
    
     "kmax" is the initial (maximum) number of mixture components
     "kmin" is the minimum number of components to be tested
    
     "covoption" controls the covarince matrix
     covoption = 0 means free covariances
     covoption = 1 means diagonal covariances
     covoption = 2 means a common covariance for all components
     covoption = 3 means a common diagonal covarince for all components
    
     "regularize" is a regularizing factor for covariance matrices; in very small samples,
     it may be necessary to add some small quantity to the diagonal of the covariances
    
     "th" is a stopping threshold
    
     Outputs:
    
     "bestk" is the selected number of components
     "bestpp" is the obtained vector of mixture probabilities
     "bestmu" contains the estimates of the means of the components
              it has bestk columns by d lines
     "bestcov" contains the estimates of the covariances of the components
               it is a three-dimensional (three indexes) array
               such that bestcov(:,:,1) is the d by d covariance of the first
               component and bestcov(:,:,bestk) is the covariance of the last
               component
    
     "dl" contains the successive values of the cost function
     "countf" is the total number of iterations performed\
     Minor modifications for using with vanishing point detection algorithm by:
     Jose Lezama
     jlezama@gmail.com
     converted by 
     Majidov Ikhtiyor
    """
    verb = 0 # verbose mode
    PLOT = 0 # 0
    bins = 40 # number of bins for the univariate histograms for visualization
    dl = [] # vector to store the consecutive values of the cost function
    dimens, npoints = np.shape(y)
    
    if(covoption == 0):
        npars = dimens + dimens*(dimens + 1)/ 2
        #this is for free covariance matrices 
    elif(covoption == 1):
        npars = 2 * dimens
        # this is for diagonal covariance matrices
    elif(covoption == 2):
        npars = dimens
        # this is for a common covariance matrix
        # independently of its structure
    elif(covoption == 3):
        npars = dimens + dimens * (dimens+1)/2
    nparsover2 = npars/2
    # we choose which axes to use in the plot 
    # in case of higher dimensional data (> 2)
    
    axis1 = 1 # i think this is prone to change
    axis2 = 2 # i think this should change
    
    # kmax is the initial number of mixture components
    k = kmax
    
    # incdic will contain the assignments of each data point to 
    # the mixture components as result of the E -step
    indic = np.zeros(k, npoints)
    semi_indic = np.zeros(k, npoints)
    
    # initialization : we will initialize the means of the k components
    # with k randomly chosen data points. Randperm(n) is a Matlab function
    # that equivialent to python np.random.permute
    # that generates random permutations of the integerers from 1 to n
    randindex = np.random.permute(npoints)
    randindex = randindex[0:k]
    estmu = y[:, randindex]
    
    
    # the initial estimations of the mixing probabilities are set to 1/k
    estpp = (1/k) * np.ones((1, k)) # 
    
    # here we compute the global covariance of the data
    globcov = np.cov(y) # numpy cov is almost equivialent to matlab cov (just use transpose of the data)
    estcov = np.array(dimens,dimens,k)
    for i in range(k):
        # the covariances are initialized to diagonal matrices proportional
        # to 1/10 of the mean variance along all the axes
        # of course, this can be changed
        estcov[:,:,i] = np.diag(np.ones(dimens)  * max(np.diag(globcov/10)))
        
    # having the intial means covariances and probabilities we can
    # initialize the indicator functions following the standard EM equation
    # Notice that these are unnormalized values
    for i in range(k):
        
        semi_indic[i] = my_multinorm(y, estmu[:,i], estcov[:, :, i], npoints)
        indic[i,:] = semi_indic[i,:] * estpp[0][i]
        
    # we can use the inic variabels (unnormalized) to compute the
    # loglikelihood and store it for later plotting its evolution
    # we also compute and store the number of components
    countf = 1
    print('here error occurs to solve the errror just define array with shape', np.shape(np.sum(np.log(np.sum(sys.float_info.min + indic)))))
    loglike[countf] = np.sum(np.log(np.sum(sys.float_info.min + indic)))
    dlength = -loglike[countf] + (nparsover2 * np.sum(np.log(estpp))) + (nparsover2 + 0.5) * k * log(npoints)
    dl[countf] = dlength
    kappas[countf] = k
    
    # the transitions vectors will store the iteration 
    # number at which components are killed
    # transitions1 stores the iterations at which componetnts are 
    # killed by the M-step, while transitions2  stores the iterations
    # at which we force components to zero
    transitions1 = []
    transitions2 = []
    
    # minimum description length seen so far, and corresponding 
    # parameter estimates
    mindl = dl[countf]
    bestpp = estpp
    bestmu = estmu
    bestcov = estcov
    bestk = k
    
    k_cont =1 # auxiliary variable for the outer loop
    while(k_cont): # the outer loop will take us down from kmax to kmin components
        cont =1 # auxilirary variable of the inner loop
        while(cont): # this inner loop is the component-wise EM algorithm with the
            # modified M-step that  can kill components
            if(verb ==1):
                # in verbose mode we keep diplaying the minimum of the 
                # mixing probability estimates to see how colse we are
                # to killing one component
                print("k=%2d, minestepp=%0.5g".format(k, np.min(estpp)))
            
            # we begin at component 1
            comp = 1
            #  .. and can only go to last component k
            # since k may change during the process we cannot use for loop
            while(comp <=k):
                # we start with the M step
                # first we cimpute normalized indicator function
                # clear indic                    
                indic = np.zeros((k, npoints))
                print('indic initialized and its shape ', indic.shape)
                indic = my_repmat(estpp.T, [1, semi_indic.shape[1]]) * semi_indic
                
                # slow original
                normindic = indic / (sys.float_info.min + my_repmat2(np.sum(indic, 0),[k, 1]))
                
                # now we perform the standard  M step fo mean and covariance 
                normalize = 1/np.sum(normindic[comp,:])
                print('normalize value', normalize)
                
                aux = my_repmat2(normindic[comp,:], [dimens, 1]) * y
                
                estmu[:, comp] = normalize * np.sum(aux, 1)
                
                estcov[:, :, comp] = normalize * (aux * y.T) - estmu[:, comp] * (estmu[:, comp]).T + regularize * np.eye(dimens)
                
                # this is the special part of the M step that is able to 
                # kill components
                estpp[comp] = max(np.sum(normindic[comp,:],0) - nparsover2,0) / npoints
                
                # this is an auxiliary variable that will be used the 
                # signal the killing of the current componetent being update d 
                killed = 0
                
                # we now have to do some book - keeping if the current component was killed 
                # that is , we have to rearrange the vectores and matrices that store the
                # parameter estimates
                if(estpp[comp] ==0 or np.isnan(estpp(comp))): # estpp(comp) is a number
                    killed = 1
                    # we also register that at the current iteration a component was killed 
                    transitions1.append(countf)
                    
                    if(comp == 1):
                        estmu = estmu[:, 1:k] # here no need to change k py[0:k] == mat[1:k] both returns k elements from beginning !!
                        estcov = estcov[:, :, 1:k]
                        estpp = estpp[1:k]
                        semi_indic = semi_indic[1:k, :]
                    else:
                        if(comp == k):
                            estmu = estmu[:,0:k-1]
                            estcov = estcov[:, :, 0:k-1]
                            estpp = estpp[0:k-1]
                            semi_indic = semi_indic[0:k-1, :]
                        else:
                            print('shu yergacha kelganingga shukur qil, endi matrixni to`g`irla mixture4.py 201 ll ')
                            estmu = np.append(estmu[:, 0:comp-1], estmu[:,comp+1:k])
                            newcov = np.zeros((dimens, dimens,k-1))
                            for kk in range(comp-1): # here comp-1 may be comp itself you should check !!
                                newcov[:, :, kk] = estcov[:, :, kk]
                            for kk in np.arange(comp+1,k):
                                newcov[:, :, kk-1] = estcov[:, :, kk]
                                
                            
                            
                            
                    
                    
                
def my_repmat2(A, siz, n):
    return np.tile(A, [siz[0],1])

def my_repmat(A, siz):
    """
     REPMAT Replicate and tile an array.
       B = repmat(A,M,N) creates a large matrix B consisting of an M-by-N
       tiling of copies of A. The size of B is [size(A,1)*M, size(A,2)*N].
       The statement repmat(A,N) creates an N-by-N tiling.
    
       B = REPMAT(A,[M N]) accomplishes the same result as repmat(A,M,N).
    
       B = REPMAT(A,[M N P ...]) tiles the array A to produce a
       multidimensional array B composed of copies of A. The size of B is
       [size(A,1)*M, size(A,2)*N, size(A,3)*P, ...].
    
       REPMAT(A,M,N) when A is a scalar is commonly used to produce an M-by-N
       matrix filled with A's value and having A's CLASS. For certain values,
       you may achieve the same results using other functions. Namely,
          REPMAT(NAN,M,N)           is the same as   NAN(M,N)
          REPMAT(SINGLE(INF),M,N)   is the same as   INF(M,N,'single')
          REPMAT(INT8(0),M,N)       is the same as   ZEROS(M,N,'int8')
          REPMAT(UINT32(1),M,N)     is the same as   ONES(M,N,'uint32')
          REPMAT(EPS,M,N)           is the same as   EPS(ONES(M,N))
    
       Example:
           repmat(magic(2), 2, 3)
           repmat(uint8(5), 2, 3)
    
       Class support for input A:
          float: double, single
    
       See also BSXFUN, MESHGRID, ONES, ZEROS, NAN, INF.
    
       Copyright 1984-2010 The MathWorks, Inc.
       $Revision: 1.17.4.17 $  $Date: 2010/08/23 23:08:12 $
       converted to python by: Majidov Ikhtiyor
    """
    if(len(A[0]) == 1):
        
        nelems = np.prod(np.array(siz, float),0)
        # Since B doesnt exist, the first statement creates a B with
        # the right size and type. THen use scalar expansion to
        # fill the array. Finally reshape to the specified size
        print("i think we have to change B like for example np.array((some_number, shape_A*)) please look at mixture4")
#        B[nelems] = A
        B = np.zeros(siz)
        if(np.all(B[0] == B[nelems-1])):
            # if B[0] is the same 
            B[:] = A
        B = np.reshape(B, siz)
    else:
#        B = A[:, np.ones(siz[1],1)]
        B =  np.tile(A, [siz[1],1]).T
    return B
         
def my_multinorm(x, m, covar, npoints):
    # evaluates a multidimensional Gauession 
    # of mean m and a covariance matrix covar 
    # at the array of points x
    #
    # converted version of (2002): Mario A. T. Figueiredo and Anil K. Jain
    # coverting author : Majidov Ikhtiyor
    
    X = covar + sys.float_info.min * np.array([[1, 0],[0,1]])
    dd = np.linalg.det(X)
    inv = np.linalg.inv(X)
    ff = ((2 * np.pi)**(-1) * ((dd)**(-0.5)))
    
    centered = (x-m*np.ones(npoints))
    y = ff * np.exp(-0.5 * np.sum(centered * (inv * centered)))
    return y

    