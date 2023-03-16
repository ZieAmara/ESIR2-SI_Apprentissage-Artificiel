
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity


def draw_decision_boundaries(model, data):
    xx, yy = np.meshgrid(
            np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max()),
            np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max()))
    zz = np.c_[xx.ravel(), yy.ravel()]
    pd_zz = pd.DataFrame(zz)
    pd_zz.columns=['X1','X2'] # avoid warning on valid feature names
    np_pred_zz= model.predict(pd_zz)
    color_map = matplotlib.colors.ListedColormap(pd.Series(['blue', 'red']))
    fig = plt.figure(figsize=  (5,5))
    fig= plt.contourf(xx, yy, np_pred_zz.reshape(xx.shape), alpha = 0.5, cmap = color_map,levels=([0, 0.5, 1]))
    #fig = plt.scatter(zz.iloc[:,0], zz.iloc[:,1], c = pred_zz, cmap = color_map, marker='+', s=70)
    fig = plt.scatter(data.iloc[:,0], data.iloc[:,1], s = 25, c = data.iloc[:,2], cmap = color_map)


def draw_predicted_probabilities(model, data):
    xx, yy = np.meshgrid(
            np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max()),
            np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max()))
    zz = np.c_[xx.ravel(), yy.ravel()]
    pd_zz = pd.DataFrame(zz)
    pd_zz.columns=['X1','X2'] # avoid warning on valid feature names
    np_prob_zz= model.predict_proba(pd_zz)[:,1] # predict_proba return 2D values (p(w1) and p(w2))
    fig = plt.figure(figsize=  (5,5))
    fig = plt.contourf(xx, yy, np_prob_zz.reshape(xx.shape), cmap = 'RdBu_r', levels=(np.linspace(0,1,100)))
    color_map = matplotlib.colors.ListedColormap(pd.Series(['blue', 'red']))
    fig = plt.scatter(data.iloc[:,0], data.iloc[:,1], s = 25, c = data.iloc[:,2], cmap = color_map)


def draw_gaussianNB_likelihood(gnb_model, data):
    # Gaussians parameters
    # class w1=0
    mu1=[gnb_model.theta_[0, 0],gnb_model.theta_[0, 1]]
    cov1=[gnb_model.sigma_[0, 0],gnb_model.sigma_[0, 1]]
    # class w2=1
    mu2=[gnb_model.theta_[1, 0],gnb_model.theta_[1, 1]]
    cov2=[gnb_model.sigma_[1, 0],gnb_model.sigma_[1, 1]]

    N=1000
    X    = np.linspace(0, 110, N)
    Y    = np.linspace(0, 110, N)
    X, Y = np.meshgrid(X, Y)
    pos  = np.dstack((X, Y))

    mg1= multivariate_normal(mu1, [[cov1[0], 0], [0, cov1[1]]])
    Z1 = mg1.pdf(pos)
    plt.contour(X, Y, Z1, cmap='Blues')
    mg2= multivariate_normal(mu2, [[cov2[0], 0], [0, cov2[1]]])
    Z2 = mg2.pdf(pos)
    plt.contour(X, Y, Z2, cmap='Reds')
    #plt.contourf(X, Y, Z1, zdir='z', offset=0, cmap=plt.cm.viridis)

    # plot all data points
    color_map = matplotlib.colors.ListedColormap(pd.Series(['blue', 'red']))
    plt.scatter(data.iloc[:,0], data.iloc[:,1], s = 10, c = data.iloc[:,2], cmap = color_map)


def draw_sk_kde_densities(bw_h,data):
    fig = plt.figure() #figsize=(5,5)
    fig.suptitle('Sklearn multivariate KDE')
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Run KernelDensity for both classes
    np_data_w1 = data[data.Y == 0].iloc[:, :-1].to_numpy()
    np_data_w2 = data[data.Y == 1].iloc[:, :-1].to_numpy()

    kde2D_w1 = KernelDensity(kernel='gaussian', bandwidth=bw_h).fit(np_data_w1)
    kde2D_w2 = KernelDensity(kernel='gaussian', bandwidth=bw_h).fit(np_data_w2)
    
    # Define limits.
    xmin, xmax=0, 110
    ymin, ymax=0, 110
    # Format data.
    x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    
    # Evaluate kernel1 in grid positions.
    Z1 = np.reshape(np.exp(kde2D_w1.score_samples(positions.T)), x.shape)
    plt.contour(x, y, Z1, cmap='Blues',levels=8)
    plt.scatter(np_data_w1[:,0], np_data_w1[:,1], s = 10, c = 'blue')
    
    # Evaluate kernel2 in grid positions.
    Z2 = np.reshape(np.exp(kde2D_w2.score_samples(positions.T)), x.shape)
    plt.contour(x, y, Z2, 5, cmap='Reds',levels=8)
    plt.scatter(np_data_w2[:,0], np_data_w2[:,1], s = 10, c = 'red')
