
# coding: utf-8

## Segmented Regression

# This is a notebook to accompany pull request `#2677`

# In[1]:


"""
Created on Sun Oct 18 23:42:04 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
import statsmodels.graphics.regressionplots as rplots
from statsmodels.regression.linear_model import OLS
from statsmodels.base._segmented import Segmented, segmented

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)


### Example 1: Two segments

# In[2]:

nobs = 100
bp_true = -0.5
sig_e = 0.01

np.random.seed(9999)
x0 = np.sort(np.random.uniform(-2, 2, size=nobs))
x0s2 = np.clip(x0 - bp_true, 0, np.inf)
beta_diff = 0.1
beta = [1, 0, beta_diff]
#exog0 = np.column_stack((np.ones(nobs), x0, x0s2))
exog0 = np.array((np.ones(nobs), x0, x0s2)).T
y_true = exog0.dot(beta) 
y = y_true + sig_e * np.random.randn(nobs)

res_oracle = OLS(y, exog0).fit()


# We use the `segmented` function to estimate the knot location by minimizing the sum of squares.

# In[3]:

mod_base = OLS(y, exog0)

res_fitted = segmented(mod_base, 1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x0, y, 'o')
ax.plot(x0, y_true, '-', label='true')
ax.plot(x0, res_oracle.fittedvalues, '-', label='oracle')
ax.plot(x0, res_fitted.fittedvalues, '-', label='best')
ax.vlines(res_fitted.knot_location, *ax.get_ylim())
ax.legend(loc='upper left')


# The location of the knot is attached to the results instance. The true knot location is at -0.5, the estimated location is close to it.

# In[4]:

res_fitted.knot_location


# The second way of estimating the knot location is by using the `Segmented` class which is more general than the function. The basic interface assumes that the columns corresponding to the extra segments or power spline basis functions have already been added. This is mostly an interface for use of Segmented to optimize parts of a given design matrix.
# 
# The second constructor is given by the `from_model` class method, which constructs the underlying basis function and appends them to the existing design matrix `exog`.
# 
# The `Segmented` class uses inplace modification which is different from the standard model classes in statsmodels, where the model only holds on to the data and all estimation results are available in the results class. The main reason for this is that `Segmented` is intended as a class that implements an algorithm that is used by standard models instead of being a full fledged model in itself. 

# In[5]:

bounds = np.percentile(x0, [10, 50, 90])
seg1 = Segmented(mod_base, x0, target_indices=[-1], bounds=bounds, degree=1)
seg1._fit_all(maxiter=10)
res_fitted1 = seg1.get_results()
res_fitted1.knot_locations


# In[6]:

mod_base0 = OLS(y, np.ones(nobs))
seg1 = Segmented.from_model(mod_base0, x0, k_knots=1, degree=1)
seg1._fit_all(maxiter=10)
res_fitted1 = seg1.get_results()
res_fitted1.knot_locations


### Example 2: Three knots "W"

# In this example the true function has a "W" shape with three knots. It illustrates that we can successfully estimate a function with several linear segments.

# In[7]:

nobs = 500
bp_true = -0.5
sig_e = 0.1

x01 = np.sort(np.random.uniform(-1.99, 1.9, size=nobs))
x0z = np.abs(x01 % 2 - 1)
beta_diff = 0.1
beta = [1, 0, beta_diff]

y_true = x0z 
y = y_true + sig_e * np.random.randn(nobs)


# First we use the `segmented` function to fit two linear segments. This partially matches one of the end segments, but has to fit a straight line through the remainder.

# In[8]:

mod_base2 = OLS(y, np.column_stack((np.ones(nobs), x01, x01)))
res_fitted2 = segmented(mod_base2, 1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x01, y, '.', alpha=0.5)
ax.plot(x01, y_true, '-', lw=2)
ax.plot(x01, res_fitted2.fittedvalues, '-', lw=2, label='best')
ax.vlines(res_fitted2.knot_location, *ax.get_ylim())


# Next, we use the model class to fit three knots which implies four segments. Given the the difference in the slope is very large and the sample size is also relatively large, we fit the true segmented regression function very closely.

# In[9]:

mod_base0 = OLS(y, np.ones(nobs))
seg3 = Segmented.from_model(mod_base0, x01, k_knots=3, degree=1)
seg3._fit_all(maxiter=10)
res_fitted_it = seg3.get_results()


# In[10]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x01, y, '.', alpha=0.5)
ax.plot(x01, y_true, '-', lw=2)
#ax.plot(x01, x0z, '-', lw=2)
ax.plot(x01, res_fitted_it.fittedvalues, '-', lw=2, label='best')
ax.vlines(res_fitted_it.knot_locations, *ax.get_ylim())


# The knot locations are attached to the results instance, but we do not have standard errors or hypothesis tests available for them.

# In[11]:

res_fitted_it.knot_locations


# In[12]:

print(res_fitted_it.summary())


# The parameterization in power splines is such that each coefficient is the increment of the slope compared to the previous segment. We can obtain the slope in a segment by cumulating the coefficients. In this example the slope alternates between approximately -1 and +1.
# However, the columns are in general not sorted by increasing knot location.

# In[13]:

res_fitted_it.params[1:].cumsum()


# In[13]:




### Example 3: Three knots - distorted "W"

# The "W" shape in the previous example is symmetric. The initial knots are allocated based on the percentiles of the underlying data which might provide a "lucky" case for the optimization. In the next example we distort the "W" shape and make it asymmetric and slightly nonlinear.

# In[14]:

nobs = 500
bp_true = -0.5
sig_e = 0.1

np.random.seed(9999)

x01 = np.sort(np.random.uniform(-1.99, 0.9, size=nobs))
x0z = np.abs(np.exp(x01 +0.5) % 2 - 1)
y_true = x0z 
y = y_true + sig_e * np.random.randn(nobs)

mod_base0 = OLS(y, np.ones(nobs))
seg3b = Segmented.from_model(mod_base0, x01, k_knots=3, degree=1)
seg3b._fit_all(maxiter=10)
res_fitted_it2 = seg3b.get_results()

seg3b_p1, r = seg3b.add_knot(maxiter=3)
res_fitted_p1 = seg3b_p1.get_results()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x01, y, '.', alpha=0.5)
ax.plot(x01, y_true, '-', lw=2)
#ax.plot(x01, x0z, '-', lw=2)
ax.plot(x01, y_true, '-', lw=2, label='true', color='b')
ax.plot(x01, res_fitted_it2.fittedvalues, '-', lw=2, label='best-iter')
ax.plot(x01, res_fitted_p1.fittedvalues, '-', lw=2, label='best-add')
ax.vlines(res_fitted_it2.knot_locations, *ax.get_ylim())
ax.legend(loc='lower left')
ax.set_title('Optimal Knot Selection')


# In[15]:

res_fitted_it2.knot_locations


# In[16]:

res_fitted_p1.knot_locations


# In this optimization the knots are not in increasing order. Obtaining the slopes in the different segments requires the cumulative sum of the sorted coefficients, which currently requires some work.

# In[17]:

order_idx = np.argsort(res_fitted_it2.knot_locations)
order_idx, res_fitted_it2.knot_locations[order_idx]


# In[18]:

coeff_sorted = res_fitted_it2.params[np.concatenate(([1], order_idx+2))]
coeff_sorted


# In[19]:

coeff_sorted.cumsum()


# The slope coefficients for the five segment regression are similar to those of the four segment regression except that the first segment is now split into a flatter and a steeper part to better match the nonlinear slope.

# In[20]:

order_idx = np.argsort(res_fitted_p1.knot_locations)
order_idx, np.array(res_fitted_p1.knot_locations)[order_idx]
res_fitted_p1.params[np.concatenate(([1], order_idx+2))].cumsum()


### Optimization and local minima

# The default optimizer is `scipy.optimize.brent` with two value brackets, left and center, which does not enforce that the optimization of a knot remains within the interval specified by the neighboring knots. This makes brent successful in my examples to quickly find the large breaks and in all three examples it is the global optimum. The second optimizer that is connected in `Segmented` is `scipy.optimize.fminbound` which is constraint to optimize a knot location within the segment boundaries. The following example shows that this local optimization is not sufficient to obtain a global optimum. (The boundary 
# 
# If the optimization fails, then it is also possible to build up the knots sequentially. Adding knots tries out to put a knot in all existing segments, and uses the best location of those. This makes the location of an added knot relative robust to local minima.
# 
# Adding knots currently always uses `brent`, `fminbound` is not connected yet. 
# 
# At the end of this section we estimate a four knot regression using `fminbound` which results in the same optimal solution as the `brent` version and as the sequential addition of the last knot. This could indicate that a local minimum with `fminbound` is the result of having too few knots. The first part of the function in nonlinear and a four knot solution is able to provide a good approximation to the nonlinear function with two segments. An optimal three knot regression has to fit the nonlinear part as a single straight line.

# In[21]:

seg3b = Segmented.from_model(mod_base0, x01, k_knots=3, degree=1)
seg3b._fit_all(maxiter=10, method='fminbound')
res_fitted_it2 = seg3b.get_results()

seg3b_p1, r = seg3b.add_knot(maxiter=10)
res_fitted_p1 = seg3b_p1.get_results()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x01, y, '.', alpha=0.5)
ax.plot(x01, y_true, '-', lw=2)
ax.plot(x01, y_true, '-', lw=2, label='true', color='b')
ax.plot(x01, res_fitted_it2.fittedvalues, '-', lw=2, label='best-iter')
ax.plot(x01, res_fitted_p1.fittedvalues, '-', lw=2, label='best-add')
ax.vlines(res_fitted_it2.knot_locations, *ax.get_ylim())
ax.legend(loc='lower left')
ax.set_title('Optimal Knot Selection')


# In[22]:

res_fitted_it2.knot_locations


# In[23]:

res_fitted_p1.knot_locations


# In[24]:

seg4b = Segmented.from_model(mod_base0, x01, k_knots=4, degree=1)
seg4b._fit_all(maxiter=10, method='fminbound')
res_fitted_it4 = seg4b.get_results()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x01, y, '.', alpha=0.5)
ax.plot(x01, y_true, '-', lw=2)
ax.plot(x01, y_true, '-', lw=2, label='true', color='b')
ax.plot(x01, res_fitted_it4.fittedvalues, '-', lw=2, label='best-iter')
ax.vlines(res_fitted_it4.knot_locations, *ax.get_ylim())
ax.legend(loc='lower left')
ax.set_title('Optimal Knot Selection')


# In[25]:

res_fitted_it4.knot_locations


### Summary and Todos

# The Segmented class implements the core algorithm for choosing knot locations. Some options are still missing or need refactoring.
# 
# The current implementation assumes that we have the full model available and that we estimate the full model in each step. Instead, we could use the projection on left out variables which would be computationally more efficient.
# 
# Sequential addition of knots will also allow us to use information criteria to choose the optimal number of knots. 
# 
# Currently there is no direct integration with the existing model infrastructure, especially connecting to the information of a dataframe or of the formulas in the original model.
# 
# Right now a predict function, or stateful transform is also missing, which is necessary for evaluating the piecewise function or generate the power spline basis for new values of the underlying explanatory variable.

# In[25]:




# In[25]:



