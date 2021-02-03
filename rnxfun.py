#rnxfun.py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
#import numpy.matlib
import matplotlib.pyplot as plt

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
    

def nx_scores(k,pts,X,Y):

    nbr = len(X) #number of colums
    nmo = nbr-1
    nmt = nbr-2

    rpt = np.floor(np.prod(Y.shape)/3) # elements in Y
    k=nbr #default value for k


    Dx = euclidean_distances(X,X)
    Dy = euclidean_distances(Y,Y)
    #creating output

    n,x,p,b = nx_trusion(coranking(Dx,Dy))
    Q_NX = n + x + p
    #B_NX = x - n
    LCMC = np.subtract(Q_NX,b)
    R_NX = np.divide(LCMC[:-1][:], 1-b[:-1][:])
    #print(n)
    #print(R_NX)

    #kavg = np.divide(np.dot(np.array(list(range(1,nmt))),R_NX),np.sum(R_NX,0))
    #pct = [5,10,25,50,75,90,95,100]
    #Rpct = np.percentile(R_NX,pct,axis=0)

    wgh = np.divide(1,np.array(list(range(1,nmo+1))))
    wgh = wgh/np.sum(wgh)
    #Qavg = wgh*Q_NX
    #Bavg = wgh*B_NX
    wgh = np.divide(1,np.array(list(range(1,nmt+1))))
    wgh = wgh/np.sum(wgh)
    #Ravg = wgh*R_NX

    #TR = np.argsort(R_NX, axis=1)
    #TR = np.argsort(TR, axis=1)
    #AR = wgh*(rpt+1-TR)
    #FS = (np.multiply(wgh,5))/(np.max(list(range(1,rpt)))*(TR-1))

    kra = nmt
    yla = '$100 R_{\mathrm{NX}}(K)$'
    yco = R_NX
    ylb = 0
    yub = 5*np.ceil(20*np.max(yco))
    lpo = 'South' # not always optimal when loab is true

    v1 = list(range(1,kra+1))
    #print(rpt)
    #print(LCMC)
    #print(v1)
    #print(yco[:][:2])
    #for t in range(int(rpt)):
    #    plt.plot(v1,yco[:,[t]])
        #plt.plot([t],[t])
    #print(rpt)
    #print(yco[:,[1]])
    plt.plot(v1,100*R_NX)
     
    plt.show()





    