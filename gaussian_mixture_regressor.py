from sklearn.base import BaseEstimator
from collections import Counter
import matplotlib.mlab as mlab
import numpy as np
from sklearn import mixture
import cluster_utils
from scipy.stats import multivariate_normal
import scipy

class GaussianMixtureRegressor(BaseEstimator):
    def __init__(self, maxClusters = 10):
        self.maxClusters = maxClusters

    def fit(self, x, y, n_components = None):
        # Find optimal number of clusters
        X = np.zeros((len(x), 2))
        X[:, 0] = x
        X[:, 1] = y
        self.maxClusters = min(len(x), self.maxClusters)
        if n_components is None:
            n_components = cluster_utils.silhouetteAnalyis(X, range(2, self.maxClusters))
        self.clf  = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
        self.clf.fit(X)
        
    def predict(self, X_pred):
        X_pred = np.array([X_pred]).ravel()[:, np.newaxis]
        n_samples, n_features_1 = X_pred.shape
        xFeatureIndices = np.array([0]) # first feature will be prediction input
        n_features_2 = self.clf.means_.shape[1] - n_features_1
        Y = np.empty((n_samples, n_features_2))

        for n in range(n_samples):
            Y[n] = self.conditionalDistribution(self.clf, X_pred[n], xFeatureIndices)
        return Y
        
    def conditionalDistribution(self, clf, X_pred, xFeatureIndices):
        n_components = clf.n_components
        n_features = clf.means_.shape[1] - len(xFeatureIndices)
        means = np.empty((n_components, n_features))
        covariances = np.empty((n_components, n_features, n_features))
        priors = np.empty(n_components)
        for k in range(n_components):
            multivariate_normal(clf.means_[k], clf.covariances_[k])
            yFeatureIndices = self.prediction_feature_indices(clf.means_[k].shape[0], xFeatureIndices)

            cov_12 = clf.covariances_[k][np.ix_(yFeatureIndices, xFeatureIndices)]
            cov_11 = clf.covariances_[k][np.ix_(yFeatureIndices, yFeatureIndices)]
            cov_22 = clf.covariances_[k][np.ix_(xFeatureIndices, xFeatureIndices)]

            prec_22 = scipy.linalg.pinvh(cov_22)
            regression_coeffs = cov_12.dot(prec_22)

            means[k] = clf.means_[k][yFeatureIndices] + regression_coeffs.dot((X_pred - clf.means_[k][xFeatureIndices]).T).T
            covariances[k] = cov_11 - regression_coeffs.dot(cov_12.T)
            pdfValue = multivariate_normal(clf.means_[k][xFeatureIndices], 
                                clf.covariances_[k][np.ix_(xFeatureIndices, xFeatureIndices)]
                                          ).pdf(X_pred)
            priors[k] = (clf.weights_[k] * pdfValue)

        factor = priors.sum()
        if factor == 0:
            factor=1
        priors /= factor
        return priors.dot(means)
    
    def prediction_feature_indices(self, n_features, x_indices):
        inv = np.ones(n_features, dtype=np.bool)
        inv[x_indices] = False
        inv, = np.where(inv)
        return inv
