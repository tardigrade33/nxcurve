#rnxfun.py
# Author: Nicolas Marin <josue.marin1729@gmail.com>
# License: MIT
from sklearn import manifold, datasets  #Datasets
import numpy as np
import rnxfun as rnx
import wrapers as drmeth

###### DATA ####################
## important variables
n_comp = 2
n_nei =12
nsamples = 500

#Swissroll
X, color = datasets.make_swiss_roll(n_samples=nsamples)
#X, color = drmeth.sphere(n_samples=nsamples)
#Scurve
#X, color = datasets.make_s_curve(n_samples=nsamples)

########## DR ##################
####Sklearn method
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=n_nei, n_components=n_comp)

####Kernel Method use wrapers KPCA
#X_r = drmeth.lle(X,n_comp,n_nei)
#drmeth.draw_projection(X,X_r,color)

rnx.nx_scores(X,X_r)