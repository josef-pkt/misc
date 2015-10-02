
# coding: utf-8

## Example: Using GMM without exog and instruments

# In the following we look at a simple example for estimating distribution parameters with Generalized Method of Moments. The difference to other examples including the IV subclasses is that we only have a single observed variable and no explanatory variables or instruments.
# 
# The main difficulty is in providing additional information to the class which cannot be inferred from the missing explanatory variables and instruments.
# 
# The example reproduces and example in Greene's Econometric Analysis textbook that is used to illustrate GMM.
# The distribution is the Gamma distribution, which can be parameterized in different ways. Greene uses four moment conditions to estimate two parameters. See Wikipedia for additional information about the distribution.

# In[1]:

from __future__ import division
import numpy as np
from scipy.special import psi
from statsmodels.sandbox.regression.gmm import GMM


# The standard way to use the GMM class is to subclass it and define a method with the moment conditions. I also add a `__init__` method to set the number of moment conditions, `k_moms`, and the number of parameters, `k_params`, that needs to be estimated. In the standard IV setting these two are inferred from the explanatory variables and the instruments. I am coming back to this below.

# In[2]:

class GMMGamma(GMM):

    def __init__(self, *args, **kwds):
        # set appropriate counts for moment conditions and parameters
        # TODO: clean up signature
        kwds.setdefault('k_moms', 4)
        kwds.setdefault('k_params', 2)
        super(GMMGamma, self).__init__(*args, **kwds)


    def momcond(self, params):
        p0, p1 = params
        endog = self.endog
        error1 = endog - p0 / p1
        error2 = endog**2 - (p0 + 1) * p0 / p1**2
        error3 = 1 / endog - p1 / (p0 - 1)
        error4 = np.log(endog) + np.log(p1) - psi(p0)
        g = np.column_stack((error1, error2, error3, error4))
        return g


# The data is taken from the example in Green, and is

# In[3]:

y = np.array([20.5, 31.5, 47.7, 26.2, 44.0, 8.28, 30.8, 17.2, 19.9, 9.96, 55.8, 25.2, 29.0, 85.5, 15.1, 28.5, 21.4, 17.7, 6.42, 84.9])


# Given the data and our model class, we can create a model instance, fit and look at the summary. Explanations will follow.:

# In[4]:

nobs = y.shape[0]
x = np.ones((nobs, 4))

model = GMMGamma(y, x, None)
beta0 = np.array([2, 0.1])
res = model.fit(beta0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
print(res.summary())


# The results agree with those in Greene's textbook at close to the print precision, 1e-4 relative tolerance.

# In[5]:

params_greene = [3.3589, 0.12449]
bse_greene = [0.449667, 0.029099]
params_greene / res.params - 1, bse_greene / res.bse - 1


# Some explanation to the above which applies to the current version of statsmodels (0.7), which will be changed for statsmodels 0.8.
# 
# The signature for GMM has `endog, exog, instruments` as required arguments. We can set instruments to None, but not exog. We need to use currently a "fake" exog to avoid an exception when the default parameter names are created.
# 
# The fit arguments:
# 
# We set maxiter to two so we get two-stage GMM. By default an identity matrix is used for the initial weight matrix in GMM. If we set `maxiter=1`, then we get the one step GMM estimator with fixed initial weight matrix.
# 
# The optimization problem is "not nice" in this case. The parameters are required to satisfy some inequality constraints (being positive) that we don't impose in our moment conditions. Additionally, some of the moment conditions are numerically not well behaved and are badly scaled. In this case the default optimizer, scipy's `bfgs` fails. To get a solution to the optimization we provide reasonably good starting parameters and use the more robust Nelder-Mead optimizer, `optim_method='nm'`.
# 
# By default GMM centers the moments to calculate the weight matrix. Greene does not use centering, so in order to replicate his results, we need to set the weight argument `wargs=dict(centered=False)`.
# 
# As mentioned, I will change the default behavior in statsmodels to make `exog` and `instruments` optional. ( https://github.com/statsmodels/statsmodels/issues/2633 ). The above without the fake exog and without the None instruments will then be the recommended usage.

### Alternative Implementation

# In[6]:

class GMMGamma2(GMM):

    def momcond(self, params):
        p0, p1 = params
        endog = self.endog
        error1 = endog - p0 / p1
        error2 = endog**2 - (p0 + 1) * p0 / p1**2
        error3 = 1 / endog - p1 / (p0 - 1)
        error4 = np.log(endog) + np.log(p1) - psi(p0)
        g = np.column_stack((error1, error2, error3, error4))
        return g


# As alternative, we only provide the moment condition in the GMM subclass, but we use correctly shaped fake exog and instruments to provide the information about the number of moment conditions and the number of parameters.

# In[7]:

z = np.ones((nobs, 4))

model2 = GMMGamma2(y, x, z)
res2 = model2.fit(beta0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
print(res2.summary())


# Another version that works around the problems even if we have incorrectly shaped exog and instruments is to explicitly specify the required extra information. This will also work in cases where we have a different number of exog and instruments than parameters and moment conditions.
# 
# That summary requires the correct length of parameter names, `xname`, even though k_params was specified, is currently a bug and will also be fixed in statsmodels.

# In[8]:

x = z = np.ones((nobs, 1))

model2 = GMMGamma2(y, x, z, k_moms=4, k_params=2)
res2 = model2.fit(beta0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
print(res2.summary(xname=['alpha', 'beta']))


### Extensions and comparison to other estimators

# In[9]:

# TODO

