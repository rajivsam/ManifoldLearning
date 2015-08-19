import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analysis_airfoil_noise():

    fp = "/home/admin123/MLExperiments/airfoil_self_noise.csv"
    df = pd.read_csv(fp)
    X = df.ix[:, 0:5]
    X = X.as_matrix()
    Y = df.ix[:, 5]
    Y = np.reshape(Y, (Y.shape[0], 1))
    
    
    return X, Y

def do_random_projections(X,Y = None):
    from sklearn import random_projection
    rp = random_projection.SparseRandomProjection(n_components=2,
                                                  random_state=93)
    X_projected = rp.fit_transform(X)

    do_plot(X_projected[:,0], X_projected[:, 1],Y)

    
    return

def do_PCA(X, Y = None):
    from sklearn import decomposition

    pca_X = decomposition.TruncatedSVD(n_components = 2).fit_transform(X)
    do_plot(pca_X[:,0], pca_X[:,1], Y)

    return

def do_plot(X1, X2, Y):

    if Y is None:
        plt.scatter(X1[:, 0], X2[:, 1])
    else:
        num_classes = max(np.unique(Y))
        colors = [ int(i % num_classes) for i in Y.ravel()]
        plt.scatter(X1, X2, c=colors)
        
    plt.show()

    return

def do_isomap(X,Y=None, num_nbrs = 5):
    from sklearn import manifold
    X_iso = manifold.Isomap(n_neighbors= num_nbrs,\
                            n_components=2).fit_transform(X)
    do_plot(X_iso[:,0], X_iso[:,1], Y)

    return

def do_LLE(X,Y, mth ='standard', n_nbrs = 5):
    from sklearn import manifold
    X_LLE = manifold.LocallyLinearEmbedding(n_neighbors = n_nbrs,\
                                            n_components=2,
                                      method=mth).fit_transform(X)

    do_plot(X_LLE[:,0], X_LLE[:,1], Y)

    return

def do_MDS(X,Y = None, iters = 100):
    from sklearn import manifold
    XMDS = manifold.MDS(n_components=2,\
                        n_init=1, max_iter=iters).fit_transform(X)
    do_plot(XMDS[:,0], XMDS[:,1], Y)

    return

def do_spectral_embedding(X,Y = None):
    from sklearn import manifold
    embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
    X_SE = embedder.fit_transform(X)

    do_plot(X_SE[:,0], X_SE[:,1], Y)

    return
    
    
    

def analysis_glass():
    fp = "glass.csv"
    df = pd.read_csv(fp)
    X = df.ix[:, 0:9]
    X = X.as_matrix()
    Y = df.ix[:, 9]
    Y = np.reshape(Y, (Y.shape[0], 1))

    return X, Y
    
    

    
