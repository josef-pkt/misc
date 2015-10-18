
# coding: utf-8

## Robust Standard Errors for discrete models and GLM

# **Josef Perktold** *All Rights Reserved*

# In this notebook we compare three types of models from statsmodels that are represent the same underlying model in some cases. The models are for binary or count response in the linear exponential family. The three models or model groups are:
# 
# - `discrete models`: Logit, Probit, Poisson and Negative Binomial in statsmodels.discrete which are standard maximum likelihood models.
# 
# - `generalized linear model (GLM)`: A general model class for one parameter distributions in the linear exponential family, which includes Binomial with Logit or Probit link, Poissson and Negative Binomial for fixed dispersion.
# 
# - `generalized estimating equations models (GEE)`: GEE covers the distributions of GLM, but allows for cluster correlation for one-way panel or cluster data. When we use the `Independence` correlation structure, then observations are treated as uncorrelated in the estimation of the parameters, which is the same as the corresponding GLM and discrete model.
# 
# The parameters of the conditional expection with distributions in the linear exponential family (LEF) can be consistently estimated even if the correlation and variances are misspecified. However, if those are misspecified, then the usual standard errors for the parameter estimates are not correct. We can use robust sandwich covariance matrices which are robust to a structure of misspecified correlation or heteroscedasticity. Overdispersion in Poisson or Binomial is a special case of misspecified variances. 
# 
# GEE does not assume that within cluster correlation is correctly specified by default. All other models report the usual "nonrobust" standard errors by default. However, we can choose a robust covariance in those models using the `cov_type` keyword in fit. These means that GLM or the discrete models with cluster robust standard errors are equivalent to GEE with independence correlation except for different small sample corrections. While GEE is specialized to one-way cluster correlation, we can also choose heteroscedasticity robust (HC), autocorrelation and heteroscedasticity robust (HAC) or panel time series robust covariance matrices when we use GLM or the discrete models.
# 
# GLM and discrete models have a large overlap of identical models, however, they differ in their approach and in the direction in which it is possible or easy to extend them. GLM come mostly from the statistics tradition, the main optimization approach is iterated weighted least square, and cover besides discrete distribution also continuous LEF distribution where the mean function can be estimated without specifying the dispersion parameter. GLM also provide a choice of nonlinearities through the specification of a link function. The discrete models in statsmodels came mostly from the econometrics tradition, they use standard Newton or Quasi-Newton optimizers. The optimizer allow easier extensions to multiparameter distributions. For example, GLM with Negative Binomial takes the dispersion parameter as given, while the NegativeBinomial model in discrete estimates the dispersion parameter jointly with the parameters for the mean function. There is a trend towards converging evolution in statsmodels so that GLM and the corresponding discrete models become more similar in the available features.
# 
# GLM, the discrete models, the linear models (OLS, WLS) and other models that use inherit the generic maximum likelihood model features share the same implementation of robust standard errors in statsmodels and have the same options. Models that need a robust covariance by default, GEE, RLM and QuantileRegression, have specialized implementation and do not follow yet the same pattern or have different choices for robust covariance matrices.
# 
# In the following we compare these models for the Poisson, Negative Binomial, and the Binomial-Logit and Binomial-Probit cases.

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
#  - [Linear Model](#Linear-Model)
# 

# **About the notebook**
# 
# This notebook does not have unique model and result names, running only parts might take the wrong instances, that is it will take the latest that has been calculated not the one that is calculated in previous cells. Use `Run All` or `Run All Above`.
# 
# - TODO: more cleanup, full imports, get only data from test module, maybe not even that. delete old `get_robustcov_results`
# - TODO: comparison for the linear models of GLM, OLS and GEE for robust standard errors is very brief
# - TODO: the GMM version should also soon be available.
# 
# History prior to inclusion in this repo:
# 
# initial script `try_robust_poisson.py` <BR/>
# notebook version in gist 

### Preliminaries

# First we need some imports. This includes importing the example data from a unittest module, which is given as `endog` and `exog` already.
# 
# The intial version, and the internal implementation uses a helper function `get_robustcov_results` that does the sandwich covariance calculations. This function is not for users anymore. The standard usage is now through the `cov_type` argument in the `fit` method of the model. Besides a simpler access to the robust covariances, this also provides a more robust code path that avoids potential conflicts in calculations based on different covariance types.

# In[1]:

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

# TODO: move to an Appendix

# the code fragility with cached attributes and retro-fitting of results

from statsmodels.base.covtype import get_robustcov_results

# Warning: why gives this different results than cov_type in fit in next cell, 
#          because bse has already been cached in print in previous cell
#res_hc0_ = cls.res1.get_robustcov_results('HC1')
get_robustcov_results(res1._results, 'HC1', use_self=True)
bse_rob = res1.bse
nobs, k_vars = mod.exog.shape
corr_fact = (nobs) / float(nobs - 1.)
# for bse we need sqrt of correction factor
corr_fact = np.sqrt(1./corr_fact)
print(res1.bse)

# create result but don't do anything before calculating robust cov, then it works
res1h = mod.fit(disp=False)
get_robustcov_results(res1h._results, 'HC1', use_self=True)
print(res1h.bse)

# this raises exception, use_self=False is not supported anymore
#res1h2 = get_robustcov_results(res1h._results, 'HC1', use_self=False)   


# The default covariance in Poisson and GLM is `cov_type="nonrobust"`, which assumes that the likelihood function, or, in LEF, that mean and covariance of the response variable are correctly specified.

# In[6]:

res1f_nr = mod.fit(disp=False)
print(res1f_nr.cov_type)
print(res1f_nr.bse)


# If the cov_type is one of the "HC" variants, then the Goddambe-Eicker-Huber-White robust standard error is calulated. This is robust to heteroscedasticity and misspecified distribution, but assumes that the observations are independently distributed or uncorrelated.
# 
# **Note**: The HCx variants are only available for the linear regression models. In other models, all HCx variants are **currently** treated in the same way and do not differ in their small sample adjustments.

# In[7]:

res1f_hc1 = mod.fit(disp=False, cov_type='HC1', cov_kwds={})
print(res1f_hc1.cov_type)
print(res1f_hc1.bse)


# In[8]:

res1f_hc1 = mod.fit(disp=False, cov_type='HC1')
print(res1f_hc1.cov_type)
print(res1f_hc1.bse)


# We can calculate cluster robust standard errors using `cov_type='cluster'` and providing the groups labels that indicate cluster membership in `cov_kwds`. `cov_kwds` needs to be a dictionary with the information and keys corresponding to the `cov_type`.

# In[9]:

res1f = mod.fit(disp=False, cov_type='cluster', cov_kwds=dict(groups=group))
print(res1f.cov_type)
print(res1f.bse)


# In[10]:

res_clu_nc = mod.fit(cov_type='cluster', cov_kwds=dict(groups=group, use_correction=False))
print(res_clu_nc.bse)


#### GLM Poisson

# GLM with family Poisson and the default link function which is `log` is equivalent to Poisson model in `discrete`. 

# In[11]:

print('\nGLM')
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson
from statsmodels.genmod import families

mod = GLM(endog, exog, family=Poisson())
#old version
#res_glm_nr = mod.fit()
#get_robustcov_results(res_glm_nr._results, 'HC1', use_self=True)

res_glm_nr = mod.fit(cov_type='nonrobust')  # this is also the default
print(res_glm_nr.cov_type)
print(res_glm_nr.bse)
print(np.sqrt(np.diag(res_glm_nr.cov_params())))  # just checking


# In[12]:


res_glm_clu = mod.fit()
get_robustcov_results(res_glm_clu._results, cov_type='cluster', groups=group, use_self=True)

print(res_glm_clu.cov_type)
print(res_glm_clu.bse)
print(np.sqrt(np.diag(res_glm_clu.cov_params())))


# In[13]:

res_glm_clu2 = mod.fit(cov_type='cluster', cov_kwds=dict(groups=group))
print(res_glm_clu2.cov_type + ' - fit')
print(res_glm_clu2.bse)
print(np.sqrt(np.diag(res_glm_clu2.cov_params())))


# In[14]:

res_glm_hc1 = mod.fit(cov_type='HC1', cov_kwds={})
print(res_glm_hc1.cov_type + ' - fit')
print(res_glm_hc1.bse)


# In[15]:

res_glm_clu_nsc = mod.fit(cov_type='cluster', cov_kwds=dict(groups=group, correction=False))
print(res_glm_clu_nsc.bse)


#### GEE Poisson

# GEE has three options for the covariance of the parameters that are specific to GEE. Covariance type 'naive' is the standard nonrobust covariance and assumes that the correlation and variances are correctly specified and is under independence identical the one of in GLM or in the corresponding discrete model. `"robust"` requests cluster robust standard errors that does not use the small sample degrees of freedom correction in the denominator that GLM and discrete models use by default. `"bias_reduced"` covariance is a cluster robust covariance that  reduces the bias in the estimate and improves the coverage and size of Wald tests in small samples.

# In[16]:

fam = Poisson() #families.Poisson()
ind = Independence()
mod1 = GEE(endog, exog, group, cov_struct=ind, family=fam)
res_gee = mod1.fit()
#print res_gee.summary()
print(res_gee.bse)


# In[17]:

res_gee_nr = mod1.fit(cov_type='naive')
print(res_gee_nr.bse)


# In[18]:

res_gee_bc = mod1.fit(cov_type='bias_reduced')
print(res_gee_bc.bse)


# The robust standard error differ from those of the cluster robust GLM or Poisson model, which are in between the robust and the bias corrected robust standard errors of GEE.

# In[19]:

print(res_glm_clu2.bse.values, 'GLM')
print(res_gee.bse.values, 'GEE robust')
print(res_gee_bc.bse.values, 'GEE bias reduced')


# The cluster robust covariance matrices in GLM and discrete models makes by default a correction for a small number of cluster. The default 'robust' standard errors in GEE do not make this adjustment which explains the scale differences
# Note that we have a very small number of clusters in this example, and any differences in small sample adjustments can have a large effect in this case.
# 
# The bias reduced robust covariance in GEE is not just a scale adjustment, and the standard errors differe in this example between 4% to 15% to the GLM cluster robust standard errors with small sample correction.

# In[20]:

# compare GEE with robust and Poisson with cluster with correction.
res_gee.bse / res_glm_clu2.bse.values


# In[21]:

# compare GEE with robust and Poisson with cluster without correction.
res_gee.bse / res_clu_nc.bse   


# In[21]:




# In[22]:

# compare GEE with bias-reduced and GLM with cluster with correction.
print(res_gee_bc.bse.values / res_glm_clu2.bse.values)


# The estimated Poisson model shows overdispersion. There is currently no option for robust covariance matrices in GLM or Poisson that only correct for over or under dispersion. The HC standard errors correct for dispersion but also for general unspecified heteroscedasticity. An estimate of the dispersion can be obtained by dividing the sum of squared pearson residuals by the number of observations minus the number of estimated parameters.

# In[23]:

np.sqrt(res_glm_clu2.pearson_chi2 / res_glm_clu2.df_resid)


# In[24]:

print(res_gee.summary())


#### Comparison Poisson

# In the following we can compare the results of the three models for cluster robust standard errors. The first table confirms that the parameter estimates are identical except for different convergence criteria in the optimization.

# In[25]:

res_all = [res1f, res_glm_clu, res_gee]
res_names = ['Poisson-cluster', 'GLM-cluster', 'GEE']

#print pd.concat((res_glm_clu.params, res_gee.params), axis=1)
print('Parameters')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'params')) for res in res_all]), 4), 
                 index=res_glm_clu.params.index, columns=res_names)


# In[26]:

print('Standard Errors')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'bse')) for res in res_all]), 4), 
                 index=res_glm_clu.params.index, columns=res_names)


### Negative Binomial

# In[27]:

print('\nNegative Binomial')
mod = smd.NegativeBinomial(endog, exog)
res_glm_nr = mod.fit(disp=False)
get_robustcov_results(res_glm_nr._results, 'HC1', use_self=True)
print(res_glm_nr.cov_type)
print(res_glm_nr.bse)
print(np.sqrt(np.diag(res_glm_nr.cov_params())))


# In[28]:

# old version TODO delete
res_glm_clu = mod.fit(disp=False)
get_robustcov_results(res_glm_clu._results, cov_type='cluster', groups=group, use_self=True)
print(res_glm_clu.cov_type)
print(res_glm_clu.bse)
print(np.sqrt(np.diag(res_glm_clu.cov_params())))


# In[29]:

res_glm_clu2 = mod.fit(disp=False, cov_type='cluster', cov_kwds=dict(groups=group))
print(res_glm_clu2.cov_type + ' - fit')
print(res_glm_clu2.bse)
print(np.sqrt(np.diag(res_glm_clu2.cov_params())))
bse_st = np.array([ 0.2721609 ,  0.09972456,  1.23357855,  0.5239233 ])
print(bse_st),
print('Stata')
print(res_glm_clu2.bse.values / bse_st)


# In[30]:

res_glm_hc1 = mod.fit(disp=False, cov_type='HC1', cov_kwds={})
print(res_glm_hc1.cov_type + ' - fit')
print(res_glm_hc1.bse)


### Logit

#### GLM Logit (Binomial)

# In[31]:

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

# In[32]:

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

# In[33]:

fam = families.Binomial()
ind = Independence()
mod1 = GEE(endog_bin, exog, group, cov_struct=ind, family=fam)
res_gee = mod1.fit() #start_params=res_glm_logit_clu.params)
#print res_gee.summary()
print(res_gee.bse)


# compare

# In[34]:

print(res_glm_logit_clu2.bse.values)
print(np.sqrt(np.diag(res_gee.cov_robust_bc))) 


#### Comparison Logit

# In[35]:

res_all = [res_logit_clu2, res_glm_logit_clu2, res_gee]
res_names = ['Logit-cluster', 'GLM-cluster', 'GEE']

#print pd.concat((res_glm_clu.params, res_gee.params), axis=1)
print('Parameters')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'params')) for res in res_all]), 4), 
                 index=res_glm_logit_clu.params.index, columns=res_names)


# In[36]:

print('Standard Errors')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'bse')) for res in res_all]), 4), 
                 index=res_glm_logit_clu.params.index, columns=res_names)


### Probit

#### Discrete Model Probit

# In[37]:

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

# In[38]:

#mod1 = GLM(endog_bin, exog, family=families.Binomial(links.CDFLink))
mod1 = GLM(endog_bin, exog, family=families.Binomial(links.probit))

res_glm_probit_clu = mod1.fit()
print(res_glm_probit_clu.cov_type + ' - fit')
print(res_glm_probit_clu.bse)
print(np.sqrt(np.diag(res_glm_probit_clu.cov_params()))) 


# In[39]:

res_glm_probit_clu2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
print(res_glm_probit_clu2.cov_type + ' - fit')
print(res_glm_probit_clu2.bse)
print(np.sqrt(np.diag(res_glm_probit_clu2.cov_params()))) 


# In[40]:

print('\nprobit params')
print(res_probit_clu.params)
print(res_glm_probit_clu.params)
print(res_probit_clu.bse)
print(res_glm_probit_clu.bse)


# In[41]:

print(pd.concat((res_probit_clu.params, res_glm_probit_clu.params), axis=1))
print(pd.concat((res_probit_clu.bse, res_glm_probit_clu.bse), axis=1))

print(pd.concat((res_probit_clu2.params, res_glm_probit_clu2.params), axis=1))
print(pd.concat((res_probit_clu2.bse, res_glm_probit_clu2.bse), axis=1))

# TODO: use summary_col to compare estimators


# There is a difference in the numerical Hessian between GLM-Probit and discrete Probit, and consequently in the standard errors. Discrete Probit uses an analytical expression for the Hessian while GLM-Probit uses the generic GLM calculation where one part is based on finite difference calculations.

# In[42]:

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

# In[43]:

fam = families.Binomial(links.probit)
ind = Independence()
mod1 = GEE(endog_bin, exog, group, cov_struct=ind, family=fam)
res_gee = mod1.fit(start_params=res_glm_probit_clu.params)
#print res_gee.summary()
print(res_gee.bse)


#### Comparison Probit

# compare

# In[44]:

print(res_glm_probit_clu2.bse.values)
print(np.sqrt(np.diag(res_gee.cov_robust_bc)))


# In[45]:

res_all = [res_probit_clu, res_glm_probit_clu, res_gee]
res_names = ['Probit-cluster', 'GLM-cluster', 'GEE']

#print pd.concat((res_glm_clu.params, res_gee.params), axis=1)
print('Parameters')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'params')) for res in res_all]), 4), 
                 index=res_probit_clu.params.index, columns=res_names)


# In[46]:

print('Standard Errors')
pd.DataFrame(np.round(np.column_stack([np.asarray(getattr(res, 'bse')) for res in res_all]), 4), 
                 index=res_probit_clu.params.index, columns=res_names)


# In[46]:




# In[46]:




### Linear Model

# this is a stump, just for a quick comparison
# 
# We compare OLS and GLM and GEE both with gaussian family and linear link, which are the defaults for GLM and GEE.
# 
# As we can see below, both nonrobust and cluster robust without correction are identical in the three models

# In[47]:

import statsmodels.regression.linear_model as linear

mod_ols = linear.OLS(endog, exog)
res_ols_nr = mod_ols.fit()
res_ols_clu_nc = mod_ols.fit(cov_type='cluster', cov_kwds=dict(groups=group, use_correction=False))

mod_glmgau = GLM(endog, exog)
res_glmgau_nr = mod_glmgau.fit()
res_glmgau_clu_nc = mod_glmgau.fit(cov_type='cluster', cov_kwds=dict(groups=group, use_correction=False))


mod_geegau = GEE(endog, exog, group, cov_struct=Independence())
res_geegau_nr = mod_geegau.fit(cov_type='naive')
res_geegau_rob = mod_geegau.fit(cov_type='robust')
res_geegau_bc = mod_geegau.fit(cov_type='bias_reduced')


# In[48]:

print(res_ols_nr.bse.values)
print(res_glmgau_nr.bse.values)
print(res_geegau_nr.bse.values)


# In[49]:

print(res_ols_clu_nc.bse.values)
print(res_glmgau_clu_nc.bse.values)
print(res_geegau_rob.bse.values)


# As reference we print just the summary for GEE which contains information about the cluster structure.

# In[50]:

print(res_geegau_rob.summary())


# In[50]:



