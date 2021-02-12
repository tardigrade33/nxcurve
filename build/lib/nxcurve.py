"""
rnxfun.py
Author: Nicolas Marin <josue.marin1729@gmail.com>
License: MIT
References:
[1] John A. Lee, Michel Verleysen.
    Quality assessment of nonlinear dimensionality reduction: 
    rank-based criteria.
    Neurocomputing, 72(7-9):1431-1443, March 2009.
[2] J. A. Lee, E. Renard, G. Bernard, P. Dupont, M. Verleysen
    Type 1 and 2 mixtures of Kullback-Leibler divergences
    as cost functions in dimensionality reduction
    based on similarity preservation
    Accepted in Neurocomputing, 2013.
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances


def coranking(hdpd,ldpd):
    """
    input:
        hdpd:   pairwisedistance matrix high dimention
        ldpd:   pairwisedistance lower dimention
    output: c
    c is the coranking matrix computed from the matrices of 
    pairwise distances of the high dimentional data and low dimentional data
    (hdpd,ldpd)

    """
    idx_hdpd = np.argsort(hdpd, axis=0) # index matrix of sorted data
    idx_ldpd = np.argsort(ldpd, axis=0)

    rows = len(hdpd)
    cols = len(hdpd[0])
 
    rank_hd = np.zeros((rows,cols)) # hd ranking matrix
    for j in range(rows):
        for i in range(cols):
            rank_hd[(idx_ldpd[i][j])][j] = i

    corank = np.zeros((rows,cols))

    # This method calculates coranking matrix more efficiently
    for j in range(rows):
        for i in range(cols):
            h=int(rank_hd[(idx_hdpd[i][j])][j])
            corank[i][h] =  corank[i][h] + 1

    c = np.array(corank)[1:,1:]   # remove first row and column     
    return c

def nx_trusion(c):
    """
    input: coranking matrix c
    output: n,x,p,b
    Computes the intrusion and extrusion rates according to the input coranking 
    matrix. The outputs n and x denote the intruction and extrusion
    rates as a function of K, the size of the K-ary neighbourhoods. 
    The outputs p and b are the rate of perfectly preserved ranks and the
    baseline that corresponds to the overlap between two random K-ary
    neighbourhoods.
    """
    rows = len(c)
    v1 = np.arange(1,rows+1)
    v2 = np.array([(item*(rows+1)) for item in v1]) # normalizing vector

    n = np.zeros(rows) # intrusions
    x = np.zeros(rows) # extrusions
    # acumulated sum from diagonal corranking matrix / v2
    p = np.cumsum(np.diagonal(c))
    p = np.divide(p,v2)
    b = np.outer(v1,[1/rows])
    b = np.ndarray.flatten(b)

    # intrusion and extrusion rates
    for k in range(1,rows):   # from one because the diagonal does not count
        n[k] = sum(c[k][0:k]) # lower triangle sum
        x[k] = sum(c[0:k,k])  # upper triangle sum
    
    # accumulation and normalization of n and x
    n = np.divide(np.cumsum(n),v2)
    x = np.divide(np.cumsum(x),v2)
    return n,x,p,b


def difrank(X,X_r):
    """
    input: X original data, X_r reduced data
    output: Matrix containing the difference between
            the ranking matrices of the original data and
            the reduced data
    """
    Dx  =  pairwise_distances(X)
    Dyx =  pairwise_distances(X_r)

    sort_HD  = np.argsort(Dx, axis=0)
    sort_LD  = np.argsort(Dyx, axis=0)

    rows = len(Dx)
    cols = len(Dx[0])

    rank_HD = np.zeros((rows,cols))
    rank_LD = np.zeros((rows,cols))

    for j in range(cols):
        for i in range(rows):
            rank_HD[(sort_HD[i][j])][j] = i
            rank_LD[(sort_LD[i][j])][j] = i

    rank_HD = np.array(rank_HD)
    rank_LD = np.array(rank_LD)

    dif =  rank_HD - rank_LD
    tmp = np.matlib.repmat((np.max(np.abs(dif),axis=0)) ,np.size(dif,axis=1),1)
    return np.divide(dif,tmp)



def quality_curve(X,X_r,n_neighbors,opt,graph=False):
    """
    input: X original data, X_r reduced data, n_neighbors, option, graph
    output: _NX vector, area under the curve and plot if graph == True
    Any character in the following list generates a new figure: (opt)
    q: Q_NX(K)
    b: N_NX(K)
    l: LCMC(K)
    r: R_NX(K)

    """
    nbr = len(X) #number of colums
    nmo = nbr-1
    nmt = nbr-2
    
    Dx = pairwise_distances(X)
    Dy = pairwise_distances(X_r)
    #calculating intrusions and extrusions
    c = coranking(Dx,Dy)
    n,x,p,b = nx_trusion(c)

    #commented code to be implemented in the future
    #kavg = np.divide(np.dot(np.array(list(range(1,nmt))),R_NX),np.sum(R_NX,0))
    #pct = [5,10,25,50,75,90,95,100]
    #Rpct = np.percentile(R_NX,pct,axis=0)
    #print(Rpct)
    #TR = np.argsort(R_NX, axis=1) #avoid last row of Q_NX and LCMC
    #TR = np.argsort(TR, axis=1) #ranks from last to first for all K
    #AR = wgh*(rpt+1-TR) #weighted average in alogplot
    #FS = (np.multiply(wgh,5))/(np.max(list(range(1,rpt)))*(TR-1)) #five stars system (0 to 5)

    #functions for every curve
    def _rnx():
        Q_NX = n + x + p
        LCMC = np.subtract(Q_NX,b)
        R_NX = np.divide(LCMC[:-1][:], np.subtract(1,b[:-1][:]))
        wgh  = np.divide(1,np.array(list(range(1,nmt+1))))
        wgh  = wgh/np.sum(wgh)
        Ravg = np.dot(wgh,R_NX) #area under R_NX in a logplot
        name = 'R_NX(K)'
        return R_NX, Ravg, name

    def _qnx():
        Q_NX = n + x + p
        wgh  = np.divide(1,np.array(list(range(1,nmo+1))))
        wgh  = wgh/np.sum(wgh)
        Qavg = np.dot(wgh,Q_NX)   #area under Q_NX in a logplot
        name = "Q_NX(K)"
        return Q_NX, Qavg, name 
    
    def _bnx():
        B_NX = np.subtract(x,n)
        wgh  = np.divide(1,np.array(list(range(1,nmo+1))))
        wgh  = wgh/np.sum(wgh)
        Bavg = np.dot(wgh, B_NX)   #area under B_NX in a logplot
        name = 'B_NX(K)'
        return B_NX, Bavg, name

    # logic to draw and return
    if opt == 'q':
        cdata, auc, name = _qnx()
        if graph:
            draw_curve(cdata, auc, name, n_neighbors)
        return cdata, auc, name
        
    elif opt == 'b':
        cdata, auc, name = _bnx()
        if graph:
            draw_curve(cdata, auc, name, n_neighbors)
        return cdata, auc, name
        
    elif opt == 'r':
        cdata, auc, name = _rnx()
        if graph:
            draw_curve(cdata, auc, name, n_neighbors)
        return cdata, auc, name
        
    else:
        raise Exception('opt shoud be one of the following [q, b, r]')

    
    




def draw_curve(curve_data, area, name, knn):
    """
    input:  curve data, area under the curve, name, nearest neighbors
    output: plot the curve
    """
    size_x = curve_data.shape[0]+1
    v1 = np.array(list(range(size_x-1)))
    #size_x = len(curve_data)
    
    plt.figure(figsize=(10, 7))

    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 

    # Limit the range of the plot to only where the data is.    
    # Avoid unnecessary whitespace.    
    plt.ylim(0, 100)    
    plt.xlim(1,size_x)

    # Make sure your axis ticks are large enough to be easily read.    
    # You don't want your viewers squinting to read your plot.    
    plt.yticks(range(0, 101, 10), [str(x) for x in range(0, 101, 10)], fontsize=14)    
    plt.xticks(fontsize=14)

    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.    
    for y in range(10, 101, 10):    
        plt.plot(range(0, size_x), [y] * len(range(0, size_x)), "--", lw=0.5, color="black", alpha=0.3) 

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")  

    #Actually ploting the data
    plt.plot(v1, 100 * curve_data, label = name)
    plt.xlabel('K', fontsize=20, color="blue")
    plt.ylabel('100 * ' + name, fontsize=20, color="blue")
    plt.xscale('log')
    plt.plot([knn,knn],[0,100])  #division line knn
    plt.text(3, 3, str(area*100), style='italic', fontsize=14, color="blue")

    plt.tight_layout()
    plt.show()

    # Finally, save the figure as a PNG.    
    # You can also save it as a PDF, JPEG, etc.    
    # Just change the file extension in this call.    
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    
    #plt.savefig("percent-bachelors-degrees-women-usa.png", bbox_inches="tight") 







