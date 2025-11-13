import os, sys
import numpy as np
import scipy.constants as sc

### User inputs
pri_types = [ 'normal',  'normal', 'uniform',  'normal', 'uniform',
              'normal', 'normal',  'normal', 'normal', 'uniform',
             'uniform', 'uniform', 'uniform', 'normal',  'normal', 'normal']
pri_pars = [ [46, 3], [157, 3], [0.1, 0.8], [110, 10], [0., 0.4],
             [1.25, 0.15], [73, 10], [-0.7, 0.1], [110, 10], [0, 50],
             [1.0, 30.0], [-7, 1], [0, 50], [3.86e3, 1.5e2], [-0.14, 0.05], [-0.52, 0.05] ]


### Pre-defined standard functions
# uniform, bounded prior: ppars = [low, hi]
def logprior_uniform(theta, ppars):
    if np.logical_and((theta >= ppars[0]), (theta <= ppars[1])):
        return np.log(1 / (ppars[1] - ppars[0]))
    else:
        return -np.inf

# Gaussian prior: ppars = [mean, std dev]
def logprior_normal(theta, ppars):
    foo = -np.log(ppars[1] * np.sqrt(2 * np.pi)) \
          -0.5 * ((theta - ppars[0])**2 / ppars[1]**2) 
    return foo

# special line-width prior
def logprior_linewidth(theta, ppars):
    lw0 = np.sqrt(2 * sc.k * ppars[1] / (28 * (sc.m_p + sc.m_e)))
    if np.logical_and((theta >= lw0), (theta <= ppars[0])):
        return 0
    else:
        return -np.inf


### Log-Prior calculator
def logprior(theta):

    # initialize
    logptheta = np.empty_like(theta)

    # user-defined calculations
    for i in range(len(theta)):
        cmd = 'logprior_'+pri_types[i]+'(theta['+str(i)+'], '+\
              str(pri_pars[i])+')'
        logptheta[i] = eval(cmd)

    # return
    return logptheta
