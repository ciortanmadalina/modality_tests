import simdkalman
import numpy as np
import pandas as pd



def getExtremePoints(data, typeOfInflexion = None, maxPoints = None):
    """
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfInflexion = None returns all inflexion points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange ==1)[0]

    if typeOfInflexion == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]
        
    elif typeOfInflexion == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif typeOfInflexion is not None:
        idx = idx[::2]
    
    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data)-1) in idx:
        idx = np.delete(idx, len(data)-1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx= idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data)//(maxPoints + 1))
    
    return idx


def kalman(train):
    # define a Kalman filter model with a weekly cycle, parametrized by a simple "smoothing factor", s
    smoothing_factor = 2

    n_seasons = 7
    # --- define state transition matrix A
    state_transition = np.zeros((n_seasons+1, n_seasons+1))
    # hidden level
    state_transition[0,0] = 1
    # season cycle
    state_transition[1,1:-1] = [-1.0] * (n_seasons-1)
    state_transition[2:,1:-1] = np.eye(n_seasons-1)

    # --- observation model H
    # observation is hidden level + weekly seasonal compoenent
    observation_model = [[1,1] + [0]*(n_seasons-1)]

    # --- noise models, parametrized by the smoothing factor
    level_noise = 0.2 / smoothing_factor
    observation_noise = 0.2
    season_noise = 1e-3

    process_noise_cov = np.diag([level_noise, season_noise] + [0]*(n_seasons-1))**2
    observation_noise_cov = observation_noise**2

    kf = simdkalman.KalmanFilter(
        state_transition,
        process_noise_cov,
        observation_model,
        observation_noise_cov)


    result = kf.compute(train, 10)
    return result.smoothed.states.mean[:,:,0]



def smooth(x,window_len=7,window='hanning'):
        if window_len<3:
                return x
#         if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#                 raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]