
# coding: utf-8

# # OLSAbsorb: Absorbing categorical variables in OLS

# One of the main usecases for this is absorbing fixed effects in panel data.
# 
# The fixed effects are included as sparse dummy matrix and partialled out of the main explanatory variables.

# In[2]:

import time

import numpy as np
from scipy import sparse
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.regression.special_linear_model import OLSAbsorb, cat2dummy_sparse
from statsmodels.tools._sparse import PartialingSparse, dummy_sparse


# In[ ]:




# In[40]:

k_cat1, k_cat2 = 500, 100

keep = (np.random.rand(k_cat1 * k_cat2) > 0.1).astype(bool)

xcat1 = np.repeat(np.arange(k_cat1), k_cat2)[keep]
xcat2 = np.tile(np.arange(k_cat2), k_cat1)[keep]
exog_absorb = np.column_stack((xcat1, xcat2))
nobs = len(xcat1)

exog_sparse = cat2dummy_sparse(exog_absorb)
beta_sparse = 1. / np.r_[np.arange(1, k_cat1), np.arange(1, k_cat2 + 1)]

np.random.seed(999)
beta_dense = np.ones(3)
exog_dense = np.column_stack((np.ones(exog_sparse.shape[0]), np.random.randn(exog_sparse.shape[0], len(beta_dense) - 1)))
y = exog_dense.dot(beta_dense) + exog_sparse.dot(beta_sparse) + 0.01 * np.random.randn(nobs)


# In[ ]:




# In[41]:

exog_absorb.shape, exog_sparse.shape, beta_sparse.shape


# In[42]:

t0 = time.time()
mod_absorb = OLSAbsorb(y, exog_dense, exog_absorb)
res_absorb = mod_absorb.fit()
t1 = time.time()
print('time: ', t1 - t0)


# In[43]:

print(res_absorb.summary())


# In[44]:

xcat1


# In[45]:

xcat2


# In[46]:

locals().keys()


# In[47]:

exog_sparse.nnz


# In[48]:

exog_sparse


# In[49]:

exog_sparse.T.dot(exog_sparse)


# In[51]:

xcat2.reshape(k_cat1, k_cat2)[:20, :20]


# In[52]:

xm = exog_dense[:,-1].reshape(k_cat1, k_cat2)


# In[53]:

xm -= xm.mean(1)[:,None]
xm -= xm.mean(0)


# In[54]:

xm[:5, :5]


# In[17]:

res_absorb.model.exog[:20, -1]


# In[18]:

xm.ravel()[:20]


# In[21]:

np.max(np.abs(xm.ravel() - res_absorb.model.exog[:, -1]))


# In[20]:

(xm.ravel(), res_absorb.model.exog[:, -1])[:20]


# In[26]:

xm = exog_dense.reshape(-1, k_cat1, k_cat2)
xm -= xm.mean(1, keepdims=True)
xm -= xm.mean(2, keepdims=True)
np.max(np.abs(xm.reshape(-1, exog_dense.shape[-1]) - res_absorb.model.exog))


# In[27]:

xm = exog_dense.reshape(-1, k_cat1, k_cat2)
xm -= np.nanmean(xm, axis=1, keepdims=True)
xm -= np.nanmean(xm, axis=2, keepdims=True)
np.max(np.abs(xm.reshape(-1, exog_dense.shape[-1]) - res_absorb.model.exog))


# In[141]:

k_cat = (k_cat1, k_cat2)
xm = exog_dense.reshape(-1, *k_cat)
for axis in range(xm.ndim):
    xm -= np.nanmean(xm, axis=axis, keepdims=True)
np.max(np.abs(xm.reshape(-1, exog_dense.shape[-1]) - res_absorb.model.exog))


# In[252]:

from pandas import DataFrame
np.random.seed(1234)
df = DataFrame({'A' : np.random.randint(0,10,size=100), 'B' : np.random.randn(100)})
df['C'] = df['B'] - df.groupby('A')['B'].transform('mean')
df.head()
df[df.A==3]


# In[253]:

df[df.A==3].B.mean()


# In[254]:

df.head()


# In[255]:

df['B'] -= df.groupby('A')['B'].transform('mean')
df.head()


# In[257]:

np.random.seed(1234)
df = DataFrame({'A' : np.random.randint(0,10,size=100), 'B' : np.random.randn(100), 'D' : np.random.randn(100)})
df[['B', 'D']] -= df.groupby('A')[['B', 'D']].transform('mean')
df[df.A==3].B.mean()


# In[258]:

df[df.A==3]


# In[268]:

np.random.seed(1234)
df = DataFrame({'A' : np.random.randint(0,10,size=100), 'B' : np.random.randn(100), 'D' : np.random.randn(100)})
df2 = df[['B', 'D']].copy()
df2[df2.columns] -= df2.groupby(df['A'])[df2.columns].transform('mean')
df2[df.A==3].B.mean()


# In[38]:

import pandas as pd
pd.__version__


# In[168]:

# with unbalanced panel

k_cat = (k_cat1, k_cat2)
xm = np.empty(exog_dense.shape[1:] + k_cat)
xm.fill(np.nan)
xm[:, xcat1, xcat2] = exog_dense.T
for it in range(3):
    for axis in range(1, xm.ndim):
        xm = xm - np.nanmean(xm, axis=axis, keepdims=True)
np.max(np.abs(xm.reshape(exog_dense.shape[-1], -1).T[keep] + exog_dense.mean(0) - res_absorb.model.wexog), axis=0)


# In[161]:

np.mean(np.abs(xm.reshape(exog_dense.shape[-1], -1).T[keep] - res_absorb.model.wexog), axis=0)


# In[165]:

exog_dense.mean(0)


# In[143]:

xm.shape


# In[144]:

xm[2, :5, :5].T


# In[148]:

xm.reshape(exog_dense.shape[-1], -1).T[:35]


# In[155]:

res_absorb.model.wexog[:5]


# In[135]:

(1 - np.isnan(xm)).sum(2)


# In[93]:

keep[:25]


# In[55]:

exog_dense.shape + k_cat


# In[ ]:




# In[59]:

xm.shape


# In[60]:

xm[:, xcat1, xcat2].shape


# In[63]:

xm.shape


# In[64]:

xm[-1, :5,:5]


# In[66]:

xm.reshape(-1, exog_dense.shape[-1])[:15]


# In[ ]:




# In[67]:

keep[:15]


# In[68]:

res_absorb.model.exog[:15]


# In[69]:

(1-keep).sum()


# In[70]:

keep.shape


# In[130]:

k_cat = (k_cat1, k_cat2)
xm = np.empty(exog_dense.shape[1:] + k_cat)
xm.fill(np.nan)
xm2 = xm.copy()
xm[:, xcat1, xcat2] = exog_dense.T
xm2[2, xcat1, xcat2] = exog_dense[:, 2]


# In[131]:

xm[:, :5, :5]


# In[114]:

np.nonzero(np.isnan(xm[2, :, :].ravel()))[0][:5]


# In[74]:

exog_dense[:15, 2]


# In[76]:

exog_dense[500:515, 2]


# In[80]:

exog_dense[xcat1 == 2][:5]


# In[83]:

np.nanmean(xm, axis=axis, keepdims=True)[:10]


# In[95]:

xcat2[:20]


# In[96]:

np.nonzero(1 - keep)[0][:5]


# In[104]:

exog_absorb[:35]


# In[119]:

xm[2, :5, :5]


# In[124]:

np.nanmean(xm2[2, :,:], axis=0, keepdims=True)[:, :5]


# In[126]:

exog_dense[xcat2 == 2, 2].mean()


# In[105]:

k_cat


# In[115]:

np.isnan(xm[2, :, :].ravel()[keep]).any()


# In[117]:

exog_dense.shape


# In[129]:

(xm2[2, :,:] - np.nanmean(xm2[2, :,:], axis=0, keepdims=True)).shape


# In[239]:

def _group_demean_iterative(exog_dense, groups, add_mean=True, max_iter=10, atol=1e-8, get_groupmeans=False):
    """iteratively demean an array for two-way fixed effects
    
    This is intended for almost balanced panels. The data is converted
    to a 3-dimensional array with nans for missing cells.
    
    currently works only for two-way effects
    groups have to be integers corresponding to range(k_cati)
    
    no input error checking
    
    This function will change as more options and special cases are
    included.
    
    Parameters
    ----------
    exog_dense : 2d ndarray
        data with observations in rows and variables in columns.
        This array will currently not be modified.
    groups : 2d ndarray, int
        groups labels specified as consecutive integers starting at zero
    max_iter : int
        maximum number of iterations
    atol : float
        tolerance for convergence. Convergence is achieved if the
        maximum absolute change (np.ptp) is smaller than atol.
        
    Returns
    -------
    ex_dm_w : ndarray
        group demeaned exog_dense array in wide format
    ex_dm : ndarray
        group demeaned exog_dense array in long format
    it : int
        number of iterations used. If convergence has not been
        achieved then it will be equal to max_iter - 1
    
    """
    # with unbalanced panel

    k_cat = tuple((groups.max(0) + 1).tolist())
    xm = np.empty(exog_dense.shape[1:] + k_cat)
    xm.fill(np.nan)
    xm[:, groups[:, 0], groups[:, 1]] = exog_dense.T
    # for final group means
    gmean = []
    if get_groupmeans:
        gmean = [np.nanmean(xm, axis=axis) for axis in range(len(k_cat))]
    keep = ~np.isnan(xm[0]).ravel()
    finished = False
    for it in range(max_iter):
        for axis in range(1, xm.ndim):
            group_mean = np.nanmean(xm, axis=axis, keepdims=True)
            xm -= group_mean
            if np.ptp(group_mean) < atol:
                finished = True
                break
        if finished:
            break
    
    xd = xm.reshape(exog_dense.shape[-1], -1).T[keep]
    if add_mean:
        xmean = exog_dense.mean(0)
        xd += xmean
        xm += xmean[:, None, None]
    return xm, xd, it

xm, xd, it = _group_demean_iterative(exog_dense, exog_absorb, max_iter=50, add_mean=False)
xm.shape, it


# In[240]:

np.max(np.abs(xm.reshape(exog_dense.shape[-1], -1).T[keep] + exog_dense.mean(0) - res_absorb.model.wexog), axis=0)


# In[212]:

np.max(np.abs(xd + exog_dense.mean(0) - res_absorb.model.wexog), axis=0)


# In[213]:

xm, xd, it = _group_demean_iterative(exog_dense, exog_absorb, max_iter=50, add_mean=True)
np.max(np.abs(xd - res_absorb.model.wexog), axis=0)


# In[214]:

xd.shape


# In[227]:

ym, yd, it = _group_demean_iterative(y[:,None], exog_absorb, max_iter=50, add_mean=True)


# In[228]:

mod_ols2 = OLS(yd, xd)
ddof = k_cat1 + k_cat2 - 2
mod_ols2.df_resid = mod_ols2.df_resid - ddof
mod_ols2.df_model = mod_ols2.df_model + ddof
res_ols2 = mod_ols2.fit()


# In[229]:

res_ols2.params


# In[230]:

res_absorb.params


# In[231]:

res_ols2.bse


# In[232]:

res_absorb.bse


# In[233]:

res_ols2.bse / res_absorb.bse


# In[234]:

ddof


# In[238]:

res_ols2.df_resid, res_absorb.df_resid, res_ols2.df_model, res_absorb.df_model


# In[236]:

print(res_ols2.summary())


# In[237]:

print(res_absorb.summary())


# In[ ]:

xx = np.array((1e4, 10))
dfxx = pd.DataFrame(xx)
dfxx.values.base is xx.base


# In[ ]:



