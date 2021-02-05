#rnxfun.py
# Author: Nicolas Marin <josue.marin1729@gmail.com>
# License: MIT

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances


def coranking(hdpd,ldpd):
    """
    input
        hdpd:   distances matrix high dimention
        ldpd:   distances matrix lower dimention
        nsamples; samples
    output
        ρij = |{k : δik < δij or (δik = δij and 1 ≤ k < j ≤ N )}|
    review that  ρij != ρik for k != j, even if δij = δik .
    """
    ndx1 = np.argsort(hdpd, axis=0)
    ndx2 = np.argsort(ldpd, axis=0)

    rows = len(hdpd)
    cols = len(hdpd[0])
 
    ndx4 = np.zeros((rows,cols))
    for j in range(rows):
        for i in range(cols):
            ndx4[(ndx2[i][j])][j] = i
    #print(rank)
    corank = np.zeros((rows,cols))
    for j in range(rows):
        for i in range(cols):
            h=int(ndx4[(ndx1[i][j])][j])
            #print(h)
            corank[i][h] =  corank[i][h] + 1
    return np.array(corank)[1:,1:]

def nx_trusion(c):
    """
    input: c matriz corank
    output; intrusiones extrusiones
    computes the intrusion and extrusion rates according to the coranking 
    matrix in c. The outputs n and x denote the intruction and extrusion
    rates as a function of K, the size of the K-ary neighbourhoods. 
    The outputs p and b are the rate of perfectly preserved ranks and the
    baseline that corresponds to the overlap between two random K-ary
    neighbourhoods.
    """
    rows = len(c)
    v1 = np.arange(1,rows+1)
    v2 = np.array([(item*(rows+1)) for item in v1])
    #print(v1)
    #print(v2)

    n = np.zeros(rows) # intrusions
    x = np.zeros(rows) # extrusions
    tmp = np.cumsum(np.diagonal(c))
    p = np.divide(tmp,v2)
    b = np.outer(v1,[1/rows])
    b = np.ndarray.flatten(b)

    for k in range(1,rows):
        n[k] = sum(c[k][0:k])
        x[k] = sum(c[0:k,k])
    
    n = np.divide(np.cumsum(n),v2)
    x = np.divide(np.cumsum(x),v2)
    return (n,x,p,b)


def difranking(X,Yr):
    Dx  =    euclidean_distances(X,X)
    Dyx =  euclidean_distances(Yr,Yr)

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
    

def nx_scores(X,Y):

    nbr = len(X) #number of colums
    nmo = nbr-1
    nmt = nbr-2
    #print(nbr)
    #rpt = np.floor(np.prod(Y.shape)/3) # columns Ya 2

    Dx = pairwise_distances(X)
    Dy = pairwise_distances(Y)
    #creating output
    n,x,p,b = nx_trusion(coranking(Dx,Dy))
    
    #quality curves
    Q_NX = n + x + p
    B_NX = x - n
    LCMC = np.subtract(Q_NX,b)
    R_NX = np.divide(LCMC[:-1][:], 1-b[:-1][:])


    #kavg = np.divide(np.dot(np.array(list(range(1,nmt))),R_NX),np.sum(R_NX,0))
    pct = [5,10,25,50,75,90,95,100]
    Rpct = np.percentile(R_NX,pct,axis=0)
    #print(Rpct)

    wgh = np.divide(1,np.array(list(range(1,nmo+1))))
    wgh = wgh/np.sum(wgh)
    #Qavg = wgh*Q_NX   #area under Q_NX in a logplot
    #Bavg = wgh*B_NX   #area under B_NX in a logplot
    wgh = np.divide(1,np.array(list(range(1,nmt+1))))
    wgh = wgh/np.sum(wgh)
    Ravg = np.dot(wgh,R_NX) #area under R_NX in a logplot
    #Ravg Es el promedio escalar de la puntuaci�n de los disttintos becindarios para R
    
    #TR = np.argsort(R_NX, axis=1) #avoid last row of Q_NX and LCMC
    #TR = np.argsort(TR, axis=1) #ranks from last to first for all K
    #AR = wgh*(rpt+1-TR) #weighted average in alogplot
    #FS = (np.multiply(wgh,5))/(np.max(list(range(1,rpt)))*(TR-1)) #five stars system (0 to 5)
    

    
    draw_curve(R_NX,Ravg,'R_NX')


def draw_curve(curve_data,area,name):
    """
    plot the curve
    """
    v1 = list(range(1,len(curve_data)+1))
    plt.plot(v1,100*curve_data)
    axes = plt.gca()
    axes.set_ylim([0,np.max(100*curve_data)+5])
    axes.set_xlim([1,len(curve_data)])
    plt.xlabel('K')
    plt.ylabel('100'+ name)
    plt.grid(True)
    plt.text(3, 3, str(area*100), style='italic',
    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.xscale('log')
    plt.yscale('linear')
    plt.yticks(list(range(0,int(5*np.ceil(20*np.max(curve_data))),5)))
    
    plt.show()



    