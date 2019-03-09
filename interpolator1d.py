from sklearn.base import BaseEstimator
from pykrige import OrdinaryKriging
import scipy.interpolate
from collections import Counter
import matplotlib.mlab as mlab

class Interpolator1D(BaseEstimator):
    def __init__(self,  interpolation_type):
        self.interpolation_type = interpolation_type

    def fit_density(self, X):
        if frequency_interpolation_type =='linear_density':
            self.model = scipy.interpolate.interp1d(
        list(Counter(X).keys()), list(Counter(X).values()),fill_value="extrapolate")

        if self.interpolation_type == 'kde':
            self.model = mlab.GaussianKDE(X, 'scott')

    def fit(self, X, y):
        if self.interpolation_type == 'kriging':
            # pykrige doesn't support 1D data for now, only 2D or 3D
            # adapting the 1D input to 2D
            self.model = OrdinaryKriging(X, np.zeros(X.shape), y, variogram_model='gaussian')
            
        if self.interpolation_type == 'linear':    
            self.model = scipy.interpolate.interp1d(X, y,fill_value="extrapolate")
            
        if self.interpolation_type =='gmm':
             self.model= cluster_utils.gmr(X, y)
        
    def predict(self, X_pred):
        if self.interpolation_type == 'kriging':
            y_pred, y_std = self.model.execute('grid', X_pred, np.array([0.]))
            return np.squeeze(y_pred)
        
        if self.interpolation_type in ['linear', 'linear_density']:
            return self.model(X_pred)
        
        if self.interpolation_type == 'kde':
            return self.model.evaluate(X_pred)
    
        if self.interpolation_type == 'gmm':
            return self.model.predict(np.array([0]), X_pred[:, np.newaxis]).ravel()




model = Interpolator1D('linear')
model.fit(X, y)
pred = model.predict(X_pred)
plt.plot(X_pred, pred)