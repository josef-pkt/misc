
# coding: utf-8

# # Robust Standard Errors for Parameters

# In the following I try to get started with a description of robust estimators for the covariance of the parameter estimates.
# 
# Author: Josef Perktold
# License: all rights reserved because I might want to write an article. Code is mostly public domain recipes for usage with statsmodels.
# 
# related notebooks, blog articles:
# 

# ## Background
# 
# # Why? Some theoretical background
# 
# The ususal standard errors for parameter estimates are derived under the assumption that the observations are independently and identically distributed (i.i.d.) or that estimates are based on a correctly specified likelihood function. This will often not be the case with actual data or with estimators other than maximum likelihood (MLE).
# 
# The simplest case is estimating with OLS when the errors are either heteroscedastic, i.e. the variance variies with the explanatory variables, or are correlated across observations, either within a cluster or autocorrelated. OLS provides a consistent estimator for the mean parameters but the standard errors for the parameters will be wrong if either heteroscedasticity or correlation is present. One solution is to use a model that explicitly takes the violation of i.i.d. into account which, however, assumes that we know what the variance function or correlation structure is. The alternative is to correct the standard errors by using an estimator for the covariance of the parameter estimates that is robust to those deviations. Specifically, we can use estimators that are robust to unspecified heteroscedasticity (HC), heteroscedasticity and autocorrelation (HAC) or to correlation of observations within a group of observations (cluster).
# 
# The estimator parameters provided by OLS is also maximum likelihood estimate under the assumption of normal distributed errors and, consequently, response. However, OLS does not assume a specific distribution to obtain consistent estimates of the mean parameters. We only need that the mean function, i.e. the linear model in this case, is correctly specified. The same property extends to all distributions that are in the linear exponential family, which includes the distribution in GLM and several distribution in discrete models, for example Probit, Logit and Poisson.
# As a consequence we can also in those models consistently estimate the mean parameters even if correlation and variance function are misspecified. However, as in the case of OLS, the usual covariance parameter is not correct with these deviation and we can use robust covariance matrices to get asymptotically correct standard errors for our parameter estimates. This case is usually refered to as Quasi Maximum Likelihood estimation (QMLE).
# Note: The term QMLE is often used in a generic way, but also refers to some specific approaches. Specifically, we will not consider the quasi likelihood as defined by ..., we will mainly focus on the estimating equations or moment conditions. This means that we are maximizing a likelihood function that only depends on the first two moments in the case of distributions in the LEF, and only the first moment, i.e. the mean function, needs to be correctly specified to obtain consistent estimators for its parameters.
# 
# The textbooks by Wooldridge and by Cameron and Trivedi provide a good introduction to QMLE from an econometrics perspective.
# 
# The second case of using sandwich covariance matrices is when we use an estimator that does not maximize a likelihood or quasi-likelihood such as M-estimators or Generalized Method of Moments, GMM, and Generalized Estimating Equations, GEE. In these case the asymptotic covariance of the parameter estimates has the sandwich form. In MLE the sandwich form reduces to the simple standard form because of the information matrix equality which implies that the inner and outer parts of the sandwich are asymptotically the same, or have the same expectation.
# 
# Statsmodels has currently M-estimators, in RLM and QuantileRegression, and moment estimators in GMM and GEE that have special forms of sandwich covariance matrices. GMM and GEE are general estimation procedures that can also reproduce MLE and QMLE estimators with simplified covariance matrices. MLE is also a special case of an M-estimator.
# 
# 

# ## Which? An overview
# 
# In the following we briefly describe the types of robust covariances that are implemented in statsmodels. I will follow a general definition for the description that is based on the moment conditions implied by an estimator.
# 
# Given a set of moment conditions, score function or estimating equations $m(\theta)$ and a matrix of derivatives of the moment conditions or Hessian $H(\theta)$, the sandwich covariance has the general form
# 
# $H^{-1} E(m m') H^{-1}$
# 
# where the inner part $V = E(m m')$ is the covariance of the moment conditions across all observations with elements $E(m_i m_j')$ for observations i and j. $m_i m_j'$ is the outer product. If all observations are independent, then $E(m_i m_j') = 0$ for $i \!= j$ which rules out correlation but still allows for unequal variances, i.e. heteroscedasticity. If all moment conditions are i.i.d., then this reduces to a variance or scale that is common to all observations.
# 
# To be more specific we can consider the linear model estimate with OLS:
# 
# The moment conditions are $m_i = u_i x_i$ where $u_i = y_i - x \beta$ are the residuals and x are exogenous explanatory variables. The derivative of the moment condition with respect to the mean parameters $beta$ for observation i is the outer product $H_i = x_i x_i'$. This also corresponds to the score and hessian functions for the normal linear model after dropping the scale or error variance which is a multiplicative factor. The variance of the moment condition between observations i and j becomes
# 
# $E(m_i m_j') = E(x_i u_i u_j x_j')$
# 
# If all observations are uncorrelated then we only have terms $E(x_i u_i u_i x_i')$, if all errors have the same variance, then this becomes the standard OLS covariance $\sigma_i * E(x_i x_i')$. In the estimator for the covariance the expectation are replaced by sums, so for the latter case we get the usual OLS inverse covariance matrix $\sigma_i * X'X$. The derivative of the OLS moment condition is also $X' X$ so we obtain the covariance matrix under i.i.d. assumption as
# 
# $\sigma_i (X' X)^{-1} (X' X) (X' X)^{-1} = \sigma_i (X' X)^{-1}$
# 
# In the following we use the following shorthand notation $M = Xu$ refers to the vector of moment conditions by observation, which has rows corresponding to the number of observations and columns corresponding to the number of moment conditions. Furthermore to save on notation, we use the same notation for true or expected values and for the estimator, $H$ then refers to the empirical Hessian or derivative of the moment condition summed over observations. All statistics are evaluated at the given parameter estimate.
# 
# 
# ### Heteroscedasticity
# 
# This estimator assumes independent or uncorrelated observations, but allows for variance that varies with explanatory variables in an unspecified way.
# 
# Because all cross terms are zero, the covariance of the moment condition becomes
# 
# $V = M' M$ in the general and $V = Xu ' Xu$ in the OLS case.
# 
# The covariance matrix of the parameter usually referred to as "HC" or "HC0" estimate is 
# 
# $H^{-1} (M' M) H^{-1}$ or $(X' X)^{-1} (M' M) (X' X)^{-1}$
# 
# This estimator can be strongly biased in small samples and various bias or small sample corrections have been proposed, those are abbreviated with "HC1", "HC2" and "HC3". Those 4 estimators are available for the linear model. There are further variations like "HC4" that are not available. Other models only have a single "HC" estimator with optional small sample correction corresponding roughly to "HC1".
# 
# (TODO details are in documentation)
# 
# In some cases the model already includes a varying variance, for example a specified heteroscedasticity in WLS or the inherent variance function in GLM/LEF models like Poisson where the variance is a function of the mean. In these cases an "HC" robust covariance matrix protects against left-over or misspecified heteroscedasticity. That mean that we obtain asymptotically correct standard errors for the mean parameter estimates even if the included variance function is not the true variance function. An application for this is overdispersion in Poisson, where the amount of overdispersion could vary with the covariates.
# 
# This is covered in all econometrics text books, details especially on small sample performance are in ...
# 
# 
# ## Cluster correlation
# 
# Often we have data that is grouped or clustered wher observations are correlated within a group or cluster but independent across clusters. Examples for this are when we observe individuals that are from the same family or location, or panel or longitudinal data with correlation across observations for the same individual but no correlation across individuals.
# 
# In this case $E(x_i u_i u_j x_j')$ is zero if the i and j are in different groups, i.e $g(i) != g(j)$ where $g(i)$ is the group indicator for observation $i$, and possibly nonzero otherwise. The estimate variance of the moment conditions aggregates over clusters:
# 
# $V = M_g' M_g$ 
# 
# where $m_g = \sum_{i:g(i)=g} x_i u_i$ and $M_g$ is the array of all $m_g$
# 
# A similar approach can be used if there are two or more cluster variables. statsmodels provides the robust covariance for one or two cluster variables.
# 
# The properties of this estimator are derived under the assumption that the number of clusters becomes large and the number of observation within clusters stays constant. If the number of clusters is small, then the estimator of the standard errors can be strongly biased. In this case a small sample correction based on the number of clusters is recommended, which is also the default in statsmodels (and in Stata).
# 
# main reference Cameron, Miller, ...
# 
# 
# 
# ## Heteroscedasticity and autocorrelation
# 
# 
# In time series the error term is often serially correlated. OLS can still provide consistent mean parameter estimates under some conditions on the explanatory variables. However, the usual OLS estimator for the standard errors will underestimate the true values if we ignore autocorrelation. 
# 
# Trying to apply the same aggregation as in the cluster case might not provide a positive definite and consistent (?check) estimate of the covariance matrix. Newey and West showed that a consistent covariance matrix can be obtained by aggregating kernel weighted observations. The estimator for the covariance of the moment conditions is in this case
# 
# $V = \sum{t} \sum{i} k(i) m_t m_{t - i}'$
# 
# where k is a kernel weight function.
# 
# (TODO: check formula)
# 
# statsmodels only implements this estimator for a given bandwidth of the kernel estimator. Optimal choice of bandwidth parameters and other approaches like withening and recoloring are not available.
# 
# This estimator is also heteroscedasticity robust. A version that is only autocorrelation robust is not available in statsmodels.
# 
# 
# ## Panel Data correlation
# 
# In cluster robust standard errors the asymptotics is for a large number of clusters for fixed number of units per group. In a second kind of panel data the number of cross sectional units stays fixed while the number of observations or time periods per unit becomes large. Many examples in finance, cross country studies or longitudinal studies follow this pattern. The observations of any unit behaves similarly to a single time series in the HAC case, but with additional contemporaneous correlation across units. This case is similar to the case with two cluster variables except that one cluster size increases in the large sample behavior. Robust standard errors combine the behavior of cluster robust standard errors for contemporaneous observations with HAC robust standard errors in the time dimension. there are two ways to sequence the aggregation along the two dimensions. The Driscoll Kraay estimator first aggregates the cross sectional dimension and applies HAC kernel aggregation on the sum of cross sectional covariances. In the second kind, we compute the HAC robust standard errors for the time series of each cross sectional unit and then aggregate over units.
# 
# The corresponding covariance types are called nw_groupsum (hac_groupsum) for Driscoll-Krasy and nw_panel (hac_panel) for the second type.
# 
# Note: inconsistent naming in statsmodels.
# 
# In this case there is no small sample correction based on the number of cross sectional units as in cluster robust standard errors. Asymptotics assume a large number of observations per unit.
# 
# (I haven't seen or looked for any study that analyses the small sample behavior of these estimators.)
# 
# main reference Peterson article
# 
# 
# 

# ## Implementation
# 
# The following collects some implementation details.
# 
# The implementation of robust covariances went through several stages of developement. The final recommended and implemented usage is as option to the `fit` methods of the models.
# 
# Besides textbooks and theoretical articles the main references for the implementation where Cameron, Miller and ... for cluster robust standard errors, Peterson and the documentation of user contributed packages in Stata for panel hac robust standard errors, and the Stata manual for the choice of small sample correction. 
# Note, the defaults, where there is overlap, are not everywhere the same as in Stata.
# Unit test are largely written against Stata for HC1, one way cluster, and HAC with OLS and against various other packages or publications for the remainder.
# 
# The keyword arguments to `fit` are `cov_type` to choose the kind of covariance matrix, `cov_kwds` to provide arguments that are either required or optional for the specific cov_type and `use_t` to choose the distribution for inference is based on the normal and chisquare or on the t and F distributions. Theoretical results are in general based on asymptotic normality, however in small samples t or F can provide in some cases a better approximation to the distribution of parameters and test statistics.
# 
# There is also a generic scaling option `scaling_factor` that can be used to adjust the covariance matrix by a given factor, see docstring below.
# 
# Some examples are
# 
#     resp = Poisson(y, x).fit(cov_type='HC1')
# 
#     firm_cluster_year_fe_ols = sm.ols('y ~ x + C(year)', df).fit(cov_type='cluster',
#                                                                  cov_kwds={'groups': df['firmid']},
#                                                                  use_t=True)
# 
#     model = ...
#     result = model.fit(cov_type='nw-panel',
#                        cov_kwds = dict(time=self.time,
#                                        maxlags=4,
#                                        use_correction='hac',
#                                        df_correction=False))
# 
# 
# 
# ### docstring
# 
# partially incorrect, outdated and incomplete
# 
# 
#     Parameters
#     ----------
#     cov_type : string
#         the type of robust sandwich estimator to use. see Notes below
#     use_t : bool
#         If true, then the t distribution is used for inference.
#         If false, then the normal distribution is used.
#     cov_kwds : depends on cov_type
#         Required or optional arguments for robust covariance calculation.
#         see Notes below
# 
#     Returns
#     -------
#     results : results instance
#         This method creates a new results instance with the requested
#         robust covariance as the default covariance of the parameters.
#         Inferential statistics like p-values and hypothesis tests will be
#         based on this covariance matrix.
# 
# now used mainly to change results instance in `Results.__init__`
# 
#     Notes
#     -----
#     Warning: Some of the options and defaults in cov_kwds may be changed in a
#     future version.
# 
#     The covariance keywords provide an option 'scaling_factor' to adjust the
#     scaling of the covariance matrix, that is the covariance is multiplied by
#     this factor if it is given and is not `None`. This allows the user to
#     adjust the scaling of the covariance matrix to match other statistical
#     packages.
#     For example, `scaling_factor=(nobs - 1.) / (nobs - k_params)` provides a
#     correction so that the robust covariance matrices match those of Stata in
#     some models like GLM and discrete Models.
# 
#     The following covariance types and required or optional arguments are
#     currently available:
# 
#     - 'HC0', 'HC1', 'HC2', 'HC3' and no keyword arguments:
#         heteroscedasticity robust covariance
#         
#  only available in linear model
#  
#  in other models only HC0 is available, the same is returned for all HCx
#         
#         
#     - 'HAC' and keywords
# 
#         - `maxlag` integer (required) : number of lags to use
#         - `kernel` string (optional) : kernel, default is Bartlett
# 
# weights_func, BUG: not connected and has no effect
#         
#         - `use_correction` bool (optional) : If true, use small sample
#               correction
# 
#     - 'cluster' and required keyword `groups`, integer group indicator
# 
#         - `groups` array_like, integer (required) :
#               index of clusters or groups
#         - `use_correction` bool (optional) :
#               If True the sandwich covariance is calulated with a small
#               sample correction.
#               If False the the sandwich covariance is calulated without
#               small sample correction.
#         - `df_correction` bool (optional)
#               If True (default), then the degrees of freedom for the
#               inferential statistics and hypothesis tests, such as
#               pvalues, f_pvalue, conf_int, and t_test and f_test, are
#               based on the number of groups minus one instead of the
#               total number of observations minus the number of explanatory
#               variables. `df_resid` of the results instance is adjusted.
#               If False, then `df_resid` of the results instance is not
#               adjusted.
# 
#     - 'hac-groupsum' Driscoll and Kraay, heteroscedasticity and
#         autocorrelation robust standard errors in panel data
#         keywords
# 
#         - `time` array_like (required) : index of time periods
#         - `maxlag` integer (required) : number of lags to use
#         - `kernel` string (optional) : kernel, default is Bartlett
#         - `use_correction` False or string in ['hac', 'cluster'] (optional) :
#               If False the the sandwich covariance is calulated without
#               small sample correction.
#               If `use_correction = 'cluster'` (default), then the same
#               small sample correction as in the case of 'covtype='cluster''
#               is used.
#         - `df_correction` bool (optional)
#               adjustment to df_resid, see cov_type 'cluster' above
#               #TODO: we need more options here
# 
#     - 'hac-panel' heteroscedasticity and autocorrelation robust standard
#         errors in panel data.
#         The data needs to be sorted in this case, the time series for
#         each panel unit or cluster need to be stacked.
#         keywords
# 
#         - `time` array_like (required) : index of time periods
# 
#         - `maxlag` integer (required) : number of lags to use
#         - `kernel` string (optional) : kernel, default is Bartlett
#         - `use_correction` False or string in ['hac', 'cluster'] (optional) :
#               If False the the sandwich covariance is calulated without
#               small sample correction.
#         - `df_correction` bool (optional)
#               adjustment to df_resid, see cov_type 'cluster' above
#               #TODO: we need more options here
# 
# kernel weights_func missing in both
# misnamed keyword nw_xxx hac_xxx
# 
# 
#     Reminder:
#     `use_correction` in "nw-groupsum" and "nw-panel" is not bool,
#     needs to be in [False, 'hac', 'cluster']
# 
#     TODO: Currently there is no check for extra or misspelled keywords,
#     except in the case of cov_type `HCx`
#     
# 
# ### some details on panel HAC estimators 
# 
# (from an answer that I wrote on the mailing list https://groups.google.com/d/msg/pystatsmodels/jRoYS5Z-S_s/WwrJ37MLAgAJ)
# 
# `hac-panel` where the keyword is actually `nw-panel` calculates the hac kernel sum for each time series defined by groups, and then aggregates, if I read the code and remember correctly.
# 
# Based on the code in regression: 
# the group_idx is internally calculated based on the time index, under the assumption that we have equal spaced time periods with no missing values in the interior (times series for individual panel units can differ in length as in unbalanced panel but only by truncation at the beginning or end).
# It looks like the time index is only used to calculate where panel units begin in the array. The time index or period labels themselves are not used.
# 
# In case of gaps:
# The idea that could be used here is to complete the panel to have no gaps, but set the weight of the filled rows to zero. I haven't tried it yet but it should work for the main results. One problem with this is that the degrees of freedom are wrong, which would have to be fixed up.
# 
# 
# `nw-groupsum` (Driscoll Kraay) uses time periods as labels to sum over all observations with the same time label, and then calculates the hac kernel over the sums for each period, assuming that the array with cross-section sums is a time series with equal spaced periods.
# 
# `nw-groupsum` (Driscoll Kraay)  can have gaps in individual timeseries, but I doubt I had used a unit test for that.
# 
# 
# cluster_2groups: this just aggregates according to the labels of the two groups.
# 
# groups for two cluster should be either a 2d array (nobs, 2) (*) or a tuple or similar with 2 1d arrays
# e.g. groups=(self.groups, self.time)
# 
# (if groups doesn't have a `shape` attribute, then we do `np.asarray(groups).T` to get the two arrays into columns. same effect as column_stack)
# 
# 
# 
# not implemented:
# unequal spaced hac plus groups:
# An *obvious* extension would have been to allow for kernels as in newey west or similar for arbitrary distance measures based on time periods interpreted in continuous time (or points in space, or any other distance measure) and allow for groups in another direction.
# This would interpret the "time" index as actual location for calculating the distance between two observations, and `groups` as index for discrete 0-1 distance.
# I gave up on implementing this because I didn't find a reference and it got a bit messy to implement. IIRC I stopped half way through implementing this generic kernel covariance.
# (Now that I think about it again, this might be a similar application as the product kernels for mixed continuous and discrete variables in kde and kernel regression.)
# 
# 
# ### HAC and kernels
# 
# there was a thread on the mailing list about missing kernel options
# https://groups.google.com/forum/#!topic/pystatsmodels/5Bz-aYeJONM
# 
# Note: `weights_func` or kernel for HAC is only the half kernel starting at lag zero and is assumed to be symmetric.
# 
# 
# ### Fixed scale
# 
# Only available for linear regression models OLS and WLS:
# 
#     'fixed scale' and optional keyword argument 'scale' which uses
#             a predefined scale estimate with default equal to one.
# 
# Similar to GLM where by default Binomial and Poisson have scale=1, but it can be changed using the `scale` argument in `GLM.fit`.

# ## Examples
# 
# I now we finally get to the examples. In the following I use only OLS to demonstrate different robust covariance matrices.
# A comparison of HC and cluster robust covariances is in this notebook http://nbviewer.jupyter.org/github/josef-pkt/misc/blob/compare_discrete_glm_cluster/notebooks/compare_discrete_glm_gee_cluster_robust.ipynb
# 
# The examples are taken mostly from the unit test suite.
# 
# We start with some imports, which include two datasets and some comparison results from other statistics packages

# In[1]:

import numpy as np
#from scipy import stats

from numpy.testing import assert_allclose


from statsmodels.regression.linear_model import OLS, WLS
#import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant

from statsmodels.datasets import macrodata, grunfeld

from statsmodels.regression.tests.results import results_macro_ols_robust as res
from statsmodels.regression.tests.results import results_grunfeld_ols_robust_cluster as res2


# ## Time Series Data

# The first dataset refers to macroeconomics data and regresses the growthrate of investment on the growth rate of gdp and on the real interest rate. The data are time series and we can use it to illustrate HC and HAC. HC applies in the same way also to cross-sectional and to clustered or panel data.

# In[2]:

d2 = macrodata.load().data
g_gdp = 400*np.diff(np.log(d2['realgdp']))
g_inv = 400*np.diff(np.log(d2['realinv']))
exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1]], prepend=False)
param_names = ['gdp', 'int', 'const']


# **nonrobust**

# In[3]:

mod_ols = OLS(g_inv, exogg)
res_ols = mod_ols.fit()
print(res_ols.summary(xname=param_names))


# **HC**

# We can compare the nonrobust, usual OLS standard errors with heteroscedasticity robust standard errors, specifically "HC1". All standard errors have increased, the increase is 15% to 50%. 

# In[4]:

res_ols_hc1 = mod_ols.fit(cov_type='HC1')
print(res_ols_hc1.summary(xname=param_names))


# In[5]:

(res_ols_hc1.bse / res_ols.bse - 1) * 100


# In[ ]:




# **HAC**
# 
# Using HAC corrects for autocorrelation as well as heteroscedasticity. Because there is no default option for maxlags, we choose 4 lags which is one year for this quarterly data. The next version shows the large sample default without small sample degrees of freedom correction and using the normal distribution for inference. In the following we ask for small sample correction and t distribution for inference. In this example there is only a small increase in the standard errors under the small sample option.
# 
# (**Note:** Stata has a `small` option with a similar effect, while the choice of distribution is specified separately from small sample corrections in statsmodels.)

# In[6]:

#res_ols_hac4 = mod_ols.fit(cov_type='HAC')   #BUG: no default for maxlags
res_ols_hac4 = mod_ols.fit(cov_type='HAC', cov_kwds={'maxlags':4}) #, 'use_correction': False})
print(res_ols_hac4.summary(xname=param_names))


# In[7]:

res_ols_hac4 = mod_ols.fit(cov_type='HAC', cov_kwds={'maxlags':4, 'use_correction': True}, use_t=True)
print(res_ols_hac4.summary(xname=param_names))


# ### Panel Data
# 
# Our second data set contains panel data with time series observed for each firm. Firms are the cross sectional units. Years are the time periods. 
# 
# We are dropping the last firm because it was missing in the Stata reference case used in the unit tests.

# In[8]:

dtapa = grunfeld.data.load_pandas()
#Stata example/data seems to miss last firm
dtapa_endog = dtapa.endog[:200]
dtapa_exog = dtapa.exog[:200]
exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)

mod_panel = OLS(dtapa_endog, exog)


# We need indicator function that define clusters, i.e. firms in this case, and time periods for use with the cluster robust and panel hac robust standard errors.
# 
# There are some restriction on the structure or stacking of the data that is largely driven by how the computations are implemented. The implementation can in some cases take advantage of the structure to provide faster computation, but overall the cases are not optimized for efficient computation. I never did any timing.
# Furthermore, much of the code has been written when similar functionality in pandas was not yet available or not yet fast compared to a pure numpy solution.
# 
# - `firm_id` is an integer group indicator, used instead of the names of firms
# - `time` is an integer index of time periods computed here by transformation of "year"
# - `tidx` is the internal index that lists the beginning and end of the observation belonging to a firm in the stacked data.
# 
# Observations for nw_panel needs to be stacked by timeseries of firms, [y_firm1, y_firm2, ....] where y_firm1 contains the timeseries of the first firm, and so on. In this example we have 10 firms and 20 time periods

# In[9]:

firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                            return_inverse=True)
groups = firm_id
#time indicator in range(max Ti)
time = np.asarray(dtapa_exog[['year']])
time -= time.min()
time = np.squeeze(time).astype(int)
# nw_panel function requires interval bounds
tidx = [(i*20, 20*(i+1)) for i in range(10)]


# In[10]:

len(firm_names), len(np.unique(time))


# In[11]:

groups


# In[12]:

time


# In[13]:

resp_nr = mod_panel.fit()
print(resp_nr.summary())


# **Note** We are just using this example to illustrate robust standard errors. The diagnostics for this model are not encouraging. The residuals seem to be far from normal distibuted, Durbin-Watson statistic indicates autocorrelation, and the condition number might be large because all variables are trending. Multicollinearity in itself does not look like a problem, all standard errors are relatively small.

# **HC**
# 
# Our first comparison is with heteroscedasticity robust standard errors, now we use "HC3" but keep using the t distribution. The largest change is for the coefficient of capital which show a large increase in the standard error and a widening of the 95% confidence interval.

# In[14]:

resp_hc = mod_panel.fit(cov_type='HC3', use_t=True)
print(resp_hc.summary())


# ### Cluster robust
# 
# Next we compute standard errors that are robust to arbitrary correlation within firms. The standard errors further increase and are now much larger than the nonrobust OLS standard errors, the standard error of the capital coefficient is more than three times the nonrobust standard error.
# 
# **Note** our sample size is relatively small, we have only 10 firms, and there could still be a relatively large small sample bias be left in these covariance estimates.
# 
# check: there was some problems with having more observations per cluster than clusters, which I don't remember.

# In[15]:

resp_clu = mod_panel.fit(cov_type='cluster', 
                         cov_kwds = dict(groups=groups,
                                         use_correction=True,
                                         use_t=True))
print(resp_clu.summary())


# In[16]:

resp_clu.bse / resp_nr.bse


# Next, we can use an analogous approach and assume that there is cross sectional correlation, because the business cycle creates correlated shocks to each firm, but assume that there is no intertemporal within correlation. In this case we can use cluster robust standard errors that defines time periods as clusters.
# 
# Under this assumption we obtain standard errors that increase only by a smaller amount over the nonrobust standard errors, and are lower than both HC3 and cluster robust to within firm correlation.

# In[17]:

resp_clut = mod_panel.fit(cov_type='cluster', 
                         cov_kwds = dict(groups=time,
                                         use_correction=True,
                                         use_t=True))
print(resp_clut.summary())


# As final version for cluster robust standard errors we can correct for two clusters, one defined by firms and the other defined by time periods. In this case we correct for within firm correlation and for contemporaneous correlation. The standard errors are approximately the same as those for only within firm correlation and slightly smaller than those.

# In[18]:

resp_clu2 = mod_panel.fit(cov_type='cluster', 
                         cov_kwds = dict(groups=(groups, time),
                                         use_correction=True,
                                         use_t=True))
print(resp_clu2.summary())


# Cluster robust standard errors take account of correlation within a cluster without regard to any sequence. In this example we do not have any information about the distance between firms in the cross sectional dimension. However, in most cases of time series serial correlation will be strong in periods that are close and not very large in periods that are distant. Furthermore, in the current sample with have more time periods than firms, so considering it as a panel with increasing time periods but fixed number of firms might be more appropriate.
# 
# In the following we compare the two panel HAC robust covariance estimators. The resulting standard errors are again larger than the nonrobust, but smaller than the standard errors that are cluster robust to within firm correlation.

# In[19]:

resp_hacp = mod_panel.fit(cov_type='nw-panel',
                          cov_kwds = dict(time=time,
                                          maxlags=4,
                                          use_correction='hac',
                                          df_correction=False),
                          use_t=True)
print(resp_hacp.summary())


# In[20]:

resp_hacg = mod_panel.fit(cov_type='nw-groupsum',
                          cov_kwds = dict(time=time,
                                          maxlags=4,
                                          use_correction='hac',
                                          df_correction=False),
                          use_t=True)
print(resp_hacg.summary())


# ### Comparison Panel robust

# In[21]:

estimators = [resp_nr, resp_hc, resp_clu, resp_clut, resp_clu2, resp_hacp, resp_hacg]
names = 'nonrobust HC3 clu-firm clu-time clu-2way hac-panel hac-group'.split()

import pandas as pd

bse_panel = pd.DataFrame([res.bse for res in estimators], columns=resp_nr.bse.index, index=names).T
bse_panel


# We see that standard errors increase to be twice or three times the nonrobust standard errors in the various versions of HAC and cluster robust standard errors.

# In[22]:

(bse_panel.T / bse_panel['nonrobust']).T    # I'm not sure why I need to transpose


# Similar to comparing standard errors across estimators and with the nonrobust as reference, we can also compare confidence intervalls implied by the different estimators. Note that I have specified to use the t distribution in all cases above, so the only difference is in the estimated covariance matrices. Instead of comparing all confidence intervals, I compute the smallest lower bound and the largest upper bound which will be the most pesimistic case based on the above estimators. 
# 
# (I am using numpy to calculate and pandas to display. It is easier for me this way.)

# In[23]:

ci_all = np.array([res.conf_int().values for res in estimators])
ci_min = ci_all[:, :, 0].min(0)
ci_max = ci_all[:, :, 1].max(0)
pd.DataFrame(np.column_stack((ci_min, ci_max)), index=bse_panel.index, columns=[0.025, 0.975])


# We can print again the nonrobust confidence intervals from above for easier comparison.

# In[24]:

resp_nr.conf_int()


# In[ ]:




# In[ ]:




# In[ ]:




# Aside: for checking usage of pandas

# In[25]:

df_bse = pd.concat([res.bse for res in estimators], axis=1)
df_bse.columns = names
df_bse


# In[26]:

# check using numpy
df_bse.values - np.column_stack([res.bse.values for res in estimators])


# In[ ]:



