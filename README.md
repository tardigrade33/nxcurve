# nxcurve

Dimensionality reduction ( DR ) is a data transformation process which provides a low-dimensional ( attribute or variable ) representation of high dimensional data sets. This with the following purposes: noise reduction, storage space reduction, data visualization, efficient data processing and to concentrate the important information in fewer variables than the original set. A performance visual measure in DM is topology preservation. [Quality curves RNX], proposed by Lee and Verleysen, evaluates performance generating a graphical representation of topology preservation.  Nowadays there is a tool for topology conservation evaluation of DM algorithms, developed also by Lee and Verleysen (2009) but this tool is implemented only in Matlab. Here a problem arises because Matlab is limited and cannot be implemented in other technologies. here, we are going to implement, in python, a software evaluation module of curves RNX in order to be used in other technologies.

## Installation

Use the package manager [pip] to install nxcurve.

```bash
pip install nxcurve
```

## Usage

```python
from sklearn import manifold, datasets  # datasets
from nxcurve import quality_curve

n_comp = 2        # number of components to be reduced
n_nei = 20        # nearest neighbors
nsamples = 2000   # number of points (samples)

# Creating manifold 
X, color = datasets.make_swiss_roll(n_samples=nsamples)

# Performing dimensionality reduction
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=n_nei, n_components=n_comp)

# Drawing RNX curve
quality_curve(X,X_r,n_nei,'r',True)

"""
    input: X original data, X_r reduced data, n_neighbors, option, graph
    output: _NX vector, area under the curve and plot if graph == True
    Any character in the following list generates a new figure: (opt)
    q: Q_NX(K)
    b: N_NX(K)
    r: R_NX(K)
"""

```

## Features
  - RNX curve and area under the curve
  - QNX curve and area under the curve
  - BNX curve and area under the curve

### Development
- Grahp for the coranking matrix
- LCMC from a coranking matrix (local continuity meta criterion)
- Error Handling

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
----

[MIT](https://choosealicense.com/licenses/mit/)

[Quality curves RNX]: <https://www.sciencedirect.com/science/article/abs/pii/S0925231213001471?via%3Dihub>
[pip]: <https://pypi.org/project/nxcurve/>
