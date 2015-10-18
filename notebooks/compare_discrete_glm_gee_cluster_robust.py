
# coding: utf-8

## Robust Standard Errors for discrete models and GLM

# **Sections**
# 
#  - [Poisson](#Poisson)
# 
#    - [Discrete Model Poisson](#Discrete-Model-Poisson)
#    - [GLM Poisson](#GLM-Poisson)
#    - [GEE Poisson](#GEE-Poisson)
#    - [Comparison Poisson](#Comparison-Poisson)
#  
#  - [Negative Binomial](#Negative-Binomial)
# 
#  - [Logit](#Logit)
# 
#    - [Discrete Model Logit](#Discrete-Model-Logit)
#    - [GLM Logit (Binomial)](#GLM-Logit-(Binomial))
#    - [GEE Logit](#GEE-Logit)
#    - [Comparison Logit](#Comparison-Logit)
# 
#  - [Probit](#Probit)
# 
#    - [Discrete Model Probit](#Discrete-Model-Probit)
#    - [GLM Probit](#GLM-Probit)
#    - [GEE Probit](#GEE-Probit)
#    - [Comparison Probit](#Comparison-Probit)
# 

# This notebook does not have unique model and result names, running only parts might take the wrong instances, that is it will take the latest that has been calculated not the one that is calculated in previous cells. Use `Run All` or `Run All Above`.
# 
# Robust bias corrected standard errors for GEE are missing in the comparison. Currently GEE does not implement default attributes based on chosen covariance type.
# 
# The cluster robust standard errors for GLM or the discrete models are in between those of the robust and the bias corrected robust standard errors in the GEE model. The standard errors of the former are proportional to the robust standard errors of GEE.
# 
# TODO: this is missing comparison for the linear models of GLM, OLS and GEE for robust standard errors.

# In[1]:


"""
Created on Mon Aug 04 18:23:10 2014

Author: Josef Perktold
converted from script try_robust_poisson.py
"""

import pandas as pd


# The following is reusing some setup code from a test module

# In[2]:

from statsmodels.discrete.tests.test_sandwich_cov import *


# Imports for Generalized Estimating Equations, GEE, which is not part of the unit test module.

# In[3]:

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence


### Poisson

#### Discrete Model Poisson

# In[4]:

res2 = results_st.results_poisson_hc1
mod = smd.Poisson(endog, exog)
res1 = mod.fit(disp=False)
print(res1.bse)


# In[5]:

from statsmodels.base.covtype import get_robustcov_results

#res_hc0_ = cls.res1.get_robustcov_results('HC1')
get_robustcov_results(res1._results, 'HC1', use_self=True)
bse_rob = res1.bse
nobs, k_vars = mod.exog.shape
corr_fact = (nobs) / float(nobs - 1.)
# for bse we need sqrt of correction factor
corr_fact = np.sqrt(1./corr_fact)


res1f = mod.fit(disp=False, cov_type='cluster', cov_kwds=dict(groups=group))
print(res1f.cov_type)
print(res1f.bse)


# In[6]:

res1f_hc1 = mod.fit(disp=False, cov_type='HC1', cov_kwds={})
print(res1f_hc1.cov_type)
print(res1f_hc1.bse)


# In[7]:

res1f_hc1 = mod.fit(disp=False, cov_type='HC1')
print(res1f_hc1.cov_type)
print(res1f_hc1.bse)


#### GLM Poisson

# In[8]:

print('\nGLM')
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson
from statsmodels.genmod import families

mod = GLM(endog, exog, family=Poisson())
res_glm_nr = mod.fit()
get_robustcov_results(res_glm_nr._results, 'HC1', use_self=True)
print(res_glm_nr.cov_type)
print(res_glm_nr.bse)
print(np.sqrt(np.diag(res_glm_nr.cov_params())))


# In[9]:

res_glm_clu = mod.fit()
get_robustcov_results(res_glm_clu._results, cov_type='cluster', groups=group, use_self=True)
print(res_glm_clu.cov_type)
print(res_glm_clu.bse)
print(np.sqrt(np.diag(res_glm_clu.cov_params())))


# In[10]:

res_glm_clu2 = mod.fit(cov_type='cluster', cov_kwds=dict(groups=group))
print(res_glm_clu2.cov_type + ' - fit')
print(res_glm_clu2.bse)
print(np.sqrt(np.diag(res_glm_clu2.cov_params())))


# In[11]:

res_glm_hc1 = mod.fit(cov_type='HC1', cov_kwds={})
print(res_glm_hc1.cov_type + ' - fit')
print(res_glm_hc1.bse)


#### GEE Poisson

# In[12]:

fam = Poisson() #families.Poisson()
ind = Independence()
mod1 = GEE(endog, exog, group, cov_struct=ind, family=fam)
res_gee = mod1.fit()
#print res_gee.summary()
print(res_gee.bse)


# The robust standard error differ from those of the cluster robust GLM or Poisson model, which are in between the robust and the bias corrected robust standard errors of GEE.

# In[13]:

print(res_glm_clu2.bse.values)
print(np.sqrt(np.diag(res_gee.cov_robust_bc)))


# It is not clear to me where the difference is comming from based on trying to reverse engineer the scaling difference between GEE robust and Poisson cluster robust standard errors. 
# Note that we have a very small number of clusters in this example, and any differences in small sample adjustments can have a large effect in this case.

# In[14]:

res_gee.bse / res_glm_clu2.bse.values


# In[15]:

np.sqrt(np.diag(res_gee.cov_robust_bc)) / res_glm_clu2.bse.values


# In[16]:

np.sqrt(4/5. * 31 / 34.), res_glm_clu2.df_resid


# In[17]:

np.sqrt(res_glm_clu2.pearson_chi2 / res_glm_clu2.df_resid)


# In[18]:

print(res_gee.summary())


#### Comparison Poisson

# In[19]:

res_all = [res1f, res_glm_clu, res_gee]
res_names = ['Poisson-cluster', 'GLM-cluster', 'GEE']

#print pd.concat((res_glm_clu.params, res_gee.params), axis=1)
print('Parameters')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'params')) for res in res_all]), 4), 
                 index=res_glm_clu.params.index, columns=res_names)


# In[20]:

print('Standard Errors')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'bse')) for res in res_all]), 4), 
                 index=res_glm_clu.params.index, columns=res_names)


### Negative Binomial

# In[21]:

print('\nNegative Binomial')
mod = smd.NegativeBinomial(endog, exog)
res_glm_nr = mod.fit(disp=False)
get_robustcov_results(res_glm_nr._results, 'HC1', use_self=True)
print(res_glm_nr.cov_type)
print(res_glm_nr.bse)
print(np.sqrt(np.diag(res_glm_nr.cov_params())))


# In[22]:

res_glm_clu = mod.fit(disp=False)
get_robustcov_results(res_glm_clu._results, cov_type='cluster', groups=group, use_self=True)
print(res_glm_clu.cov_type)
print(res_glm_clu.bse)
print(np.sqrt(np.diag(res_glm_clu.cov_params())))


# In[23]:

res_glm_clu2 = mod.fit(disp=False, cov_type='cluster', cov_kwds=dict(groups=group))
print(res_glm_clu2.cov_type + ' - fit')
print(res_glm_clu2.bse)
print(np.sqrt(np.diag(res_glm_clu2.cov_params())))
bse_st = np.array([ 0.2721609 ,  0.09972456,  1.23357855,  0.5239233 ])
print(bse_st),
print('Stata')
print(res_glm_clu2.bse.values / bse_st)

res_glm_hc1 = mod.fit(disp=False, cov_type='HC1', cov_kwds={})
print(res_glm_hc1.cov_type + ' - fit')
print(res_glm_hc1.bse)


### Logit

#### GLM Logit (Binomial)

# In[24]:

print('\nGLM Logit')
endog_bin = (endog > endog.mean()).astype(int)
mod1 = GLM(endog_bin, exog,
                   family=families.Binomial())

res_glm_logit_clu = mod1.fit()
print(res_glm_logit_clu.cov_type + ' - fit')
print(res_glm_logit_clu.bse)

res_glm_logit_clu2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
print(res_glm_logit_clu2.cov_type + ' - fit')
print(res_glm_logit_clu2.bse)
print(np.sqrt(np.diag(res_glm_logit_clu2.cov_params())))    


#### Discrete Model Logit

# Note: copy-paste results names are wrong

# In[25]:

print('\nLogit')
endog_bin = (endog > endog.mean()).astype(int)
mod1 = smd.Logit(endog_bin, exog)

res_logit_clu = mod1.fit()
print(res_logit_clu.cov_type + ' - fit')
print(res_logit_clu.bse)

res_logit_clu2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
print(res_logit_clu2.cov_type + ' - fit')
print(res_logit_clu2.bse)
print(np.sqrt(np.diag(res_logit_clu2.cov_params())))   


#### GEE Logit

# In[26]:

fam = families.Binomial()
ind = Independence()
mod1 = GEE(endog_bin, exog, group, cov_struct=ind, family=fam)
res_gee = mod1.fit() #start_params=res_glm_logit_clu.params)
#print res_gee.summary()
print(res_gee.bse)


# compare

# In[27]:

print(res_glm_logit_clu2.bse.values)
print(np.sqrt(np.diag(res_gee.cov_robust_bc))) 


#### Comparison Logit

# In[28]:

res_all = [res_logit_clu2, res_glm_logit_clu2, res_gee]
res_names = ['Logit-cluster', 'GLM-cluster', 'GEE']

#print pd.concat((res_glm_clu.params, res_gee.params), axis=1)
print('Parameters')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'params')) for res in res_all]), 4), 
                 index=res_glm_logit_clu.params.index, columns=res_names)


# In[29]:

print('Standard Errors')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'bse')) for res in res_all]), 4), 
                 index=res_glm_logit_clu.params.index, columns=res_names)


### Probit

#### Discrete Model Probit

# In[30]:

print('\nProbit')
endog_bin = (endog > endog.mean()).astype(int)
mod1 = smd.Probit(endog_bin, exog)

res_probit_clu = mod1.fit()
print(res_probit_clu.cov_type + ' - fit')
print(res_probit_clu.bse)

res_probit_clu2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
print(res_probit_clu2.cov_type + ' - fit')
print(res_probit_clu2.bse)
print(np.sqrt(np.diag(res_probit_clu2.cov_params()))) 

res_probit_hac = mod1.fit(cov_type='HAC', cov_kwds=dict(maxlags=3))
print(res_probit_hac.cov_type + ' - fit')
print(res_probit_hac.bse)
print(np.sqrt(np.diag(res_probit_hac.cov_params())))


### GLM Probit

# In[31]:

#mod1 = GLM(endog_bin, exog, family=families.Binomial(links.CDFLink))
mod1 = GLM(endog_bin, exog, family=families.Binomial(links.probit))

res_glm_probit_clu = mod1.fit()
print(res_glm_probit_clu.cov_type + ' - fit')
print(res_glm_probit_clu.bse)
print(np.sqrt(np.diag(res_glm_probit_clu.cov_params()))) 


# In[32]:

res_glm_probit_clu2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
print(res_glm_probit_clu2.cov_type + ' - fit')
print(res_glm_probit_clu2.bse)
print(np.sqrt(np.diag(res_glm_probit_clu2.cov_params()))) 


# In[33]:

print('\nprobit params')
print(res_probit_clu.params)
print(res_glm_probit_clu.params)
print(res_probit_clu.bse)
print(res_glm_probit_clu.bse)


# In[34]:

print(pd.concat((res_probit_clu.params, res_glm_probit_clu.params), axis=1))
print(pd.concat((res_probit_clu.bse, res_glm_probit_clu.bse), axis=1))

print(pd.concat((res_probit_clu2.params, res_glm_probit_clu2.params), axis=1))
print(pd.concat((res_probit_clu2.bse, res_glm_probit_clu2.bse), axis=1))

# TODO: use summary_col to compare estimators


# There is a difference in the numerical Hessian between GLM-Probit and discrete Probit, and consequently in the standard errors. Discrete Probit uses an analytical expression for the Hessian while GLM-Probit uses the generic GLM calculation where one part is based on finite difference calculations.

# In[35]:

#res_probit_clu = res_probit_clu._results  # not wrapped
res_glm_probit_clu = res_glm_probit_clu._results
sc_probit = res_probit_clu.model.jac(res_probit_clu.params)
sc_glm_probit = res_glm_probit_clu.model.score_obs(res_probit_clu.params)
print(np.max(np.abs(sc_probit - sc_glm_probit)))
hess_glm_probit = res_glm_probit_clu.model.hessian(res_probit_clu.params)
hess_probit = res_probit_clu.model.hessian(res_probit_clu.params)
print(np.max(np.abs(hess_probit - hess_glm_probit)))
print(hess_probit / hess_glm_probit)


#### GEE Probit

# In[36]:

fam = families.Binomial(links.probit)
ind = Independence()
mod1 = GEE(endog_bin, exog, group, cov_struct=ind, family=fam)
res_gee = mod1.fit(start_params=res_glm_probit_clu.params)
#print res_gee.summary()
print(res_gee.bse)


#### Comparison Probit

# compare

# In[37]:

print(res_glm_probit_clu2.bse.values)
print(np.sqrt(np.diag(res_gee.cov_robust_bc)))


# In[38]:

res_all = [res_probit_clu, res_glm_probit_clu, res_gee]
res_names = ['Probit-cluster', 'GLM-cluster', 'GEE']

#print pd.concat((res_glm_clu.params, res_gee.params), axis=1)
print('Parameters')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'params')) for res in res_all]), 4), 
                 index=res_probit_clu.params.index, columns=res_names)


# In[39]:

print('Standard Errors')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'bse')) for res in res_all]), 4), 
                 index=res_probit_clu.params.index, columns=res_names)


# In[39]:



