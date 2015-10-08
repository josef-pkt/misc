
# coding: utf-8

## GLM as a GMM Model

# The purpose of this notebook is to replicate generalized linear models, or generalized estimating equations with a independence correlation structure, as a Generalized Methods of Moments model.
# 
# The second part compares a score test for variable addition with a test of overidentifying restriction which should eventually provide a generic framework of conditional moment tests. 
# 
# **Status**: We can replicate GLM by GMM, but the rest is experimental

# In[1]:

from __future__ import division
import numpy as np
from statsmodels.sandbox.regression.gmm import GMM


# In[2]:

class GMMGLM(GMM):

    def __init__(self, endog, exog, instrument=None, glm_model=None):
        if instrument is None:
            instrument = exog
        super(GMMGLM, self).__init__(endog, exog, instrument)
        self.glm_model = glm_model
        
    def momcond(self, params):
        mom1 = self.glm_model.score_factor(params)[:, None] * self.instrument
        return mom1


# Set up a Logit model for reference. We use an example from the statsmodels documentation.

# In[3]:

import statsmodels.api as sm


# In[4]:

# stardata is binomial count we 2d endog, Doesn't work yet.

data = sm.datasets.star98.load()
data_exog = sm.add_constant(data.exog, prepend=True)[:, :8]


# Note: The affairs dataset is most likely to large to show problems with hypothesis testing that we have in small samples. The p-values of wald and score tests for testing that education has no effect are very close to each other. Asymptotically they have the same distribution, but in small samples waldtest is often liberal and overrejects, while score tests can be conservative and underreject. If both agree, then it is a strong indication that we don't have small sample problems by using the asymptotic distribution.
# 
# I'm planning to redo the analysis with a random subsample of the dataset.

# In[5]:

dta = sm.datasets.fair.load_pandas().data
dta['affair'] = (dta['affairs'] > 0).astype(float)

affair_mod = sm.formula.logit("affair ~ occupation + educ + occupation_husb" 
                   "+ rate_marriage + age + yrs_married + children"
                   " + religious", dta).fit()

# leave formulas and pandas for later
data_endog = affair_mod.model.endog
data_exog = affair_mod.model.exog

glm_binom = sm.GLM(data_endog, data_exog, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())


# In[6]:

res_binom_hc0 = glm_binom.fit(cov_type='HC0')
print(res_binom_hc0.bse)


# In[7]:

mod = GMMGLM(data_endog, data_exog, glm_model=glm_binom)
res_gmm = mod.fit(res.params*0.5, maxiter=1)#, optim_method='nm')


# In[8]:

print(res_gmm.params)


# In[9]:

print(res_gmm.summary())


# In[10]:

mod.momcond(res.params).mean(0)


# In[11]:

res.model.score(res.params)


# GMM produces the same result as GLM-Logit with robust standard errors

# In[12]:

res_gmm.bse / res_binom_hc0.bse - 1


# In[13]:

res_gmm.bse - res_binom_hc0.bse


# In[13]:




### Application: score test for added variable

# This is experimental and might not work yet.
# 
# The third variable `educ` has a p-value of 0.012 in the wald t_test based on the asymptotic normal distribution using the heteroscedasticity robust standard errors. Since wald test are in many cases liberal, we can compare this with a score test which has in many cases more accurate p-values. For simplicity we just want to test the null hypothesis that the effect of education is zero.
# 
# In the following we drop the variable from the estimated model and calculate score and conditional moment tests. In the GMM version we estimate the reduced model with `educ` as additional instrument, and look at the test for overidentifying restrictions.
# 
# The following will be easier to do with formulas and pandas DataFrames, but I want to minimize possible sources of errors, and I'm more comfortable with numpy.

# In[14]:

affair_mod.model.data.xnames


# In[15]:

idx = list(range(len(res.params)))
del idx[2]
print(res.params[idx])  # check that we don't have -0.0392 at index 2 anymore


# In[16]:

exog_reduced = data_exog[:, idx]   # exog without educ


# In[17]:

glm_binom2 = sm.GLM(data_endog, exog_reduced, family=sm.families.Binomial())
res2 = glm_binom2.fit()


# In[18]:

res2.model.score_test(res2.params, exog_extra = data_exog[:, 2])


# The pvalue of the score test, 0.01123, is close to the pvalue of the robust wald test and almost identical to the pvalue of the nonrobust wald test.

# In[19]:

res.pvalues[2]


# Next we try the test for overidentifying restriction in GMM

# In[20]:

mod_red = GMMGLM(data_endog, exog_reduced, data_exog, glm_model=glm_binom2)
res_gmm_red = mod_red.fit(res2.params*0.5, maxiter=1)
print(res_gmm_red.summary())


# In[21]:

res_gmm_red.jtest()


# The overidentifying restrictions are not rejected. 
# Note: This is a bit of a weird test in this case. We have the reduced model in the prediction, but the extra variable as an instrument. 
#     I'm not sure what we are actually testing, besides some general orthogonality condition.
#     
# What we would like to do right now is to use the restricted parameters, so we can calculate the moment conditions in the constrained model. I don't see yet where this is supported or how to support it. 
# 
# The following are just some additional checks with respect to changes in the number of GMM iterations. The above uses the identity weight matrix which is irrelevant in the exactly identified case that we had before. Now, we are in the overidentified case and need to adjust the weight matrix. As it turns out below, using two or more GMM iterations reduces the p-value for the j-test to 0.011 which is almost exactly the same as the non-robust Wald and the score test. (I need to think.)

# In[22]:

res2.params


# In[23]:

res_gmm_red0 = mod_red.fit(res2.params*0.5, maxiter=0)
res_gmm_red2 = mod_red.fit(res2.params*0.5, maxiter=2)
print(res_gmm_red0.params)
print(res_gmm_red2.params)


# In[24]:

res2.params*0.5  # check how much params moved with maxiter=0


# In[25]:

res_gmm_red2.jtest()


# In[25]:




# In[26]:

res_gmm_red10 = mod_red.fit(res2.params*0.5, maxiter=10, optim_args=dict(disp=0)) # turn of the display of iteration results
res_gmm_red10.jtest()


# In[27]:

res_gmm_red10.params


# In[28]:

res2.params


# This looks like the expected results after all. After 10 GMM iterations the parameter estimates get close to the constrained GLM estimates.
# 
# We are getting closer but this still solves a different problem since we don't keep the parameters fixed at the constrained solution.
# 
# More to come: In another branch I have generic standalone conditional moment tests, that are similar to the GMM setup but use fixed parameters estimated in the constrained model.

# Using another option, we can get the constrained parameter estimate by assigning zero weight to the extra restriction. The parameter estimates are close to the constrained GLM estimates. 
# But I never tried this before and I'm not sure whether the weights are inverse weights in this example. Also, since the estimation is not based on the efficient weights matrix, the jtest does not apply, even though the result is still available.

# In[29]:

inv_weights = np.eye(9)
inv_weights[2,2] = 0
res_gmm_red1w = mod_red.fit(res2.params*0.5, maxiter=1, inv_weights=inv_weights, optim_args=dict(disp=0), has_optimal_weights=False)
print(res_gmm_red1w.params)
print(res_gmm_red1w.jtest())


# **Conclusion** so far is that I don't know yet how to get the equivalent of an score test for a constrained model with GMM

### Lagrange Multiplier

# (experimental)
# 
# In the following we add a slack parameter to a moment condition. This is a way to estimate the parameters of the constrained model but we have the moment conditions available for the full model. See McFadden or ... (which textbook).
# 
# The wald test on the slack parameter provides a version of the Lagrange multiplier or score test. The difference to the likelihood based score test is that this moment test is based on a robust covariance matrix, in this case HC0.
# 
# Note McFadden, or Newey and McFadden (?) have several definitions of LM or score tests. I'm not sure yet which variant this is.
# 
# In this case we cannot use the test for overidentifying restrictions anymore because we added artificial parameters that makes the problem exactly identified, given that we started out with an exactly identified model. 
# 
# The test is the same variable addition test as in the previous section. The second explanatory variable is removed from the mean function of the model but included as instrument in the moment condition. The null hypothesis is that this variable has a coefficient of zero, and  is therefore orthogonal to the weighted residuals or elementary zero function of the model. ("elementary zero function" is terminology in  Davidson, MacKinnon.)

# In[29]:




# In[30]:

class GMMGLMVADD(GMM):

    def __init__(self, endog, exog, instrument=None, glm_model=None, slack_idx=None):
        if instrument is None:
            instrument = exog
        super(GMMGLMVADD, self).__init__(endog, exog, instrument)
        self.glm_model = glm_model
        
        if slack_idx is None:
            self.k_slack = 0
        else:
            self.slack_idx = slack_idx
            self.k_slack = len(slack_idx)
        
    def momcond(self, params):
        k_slack = self.k_slack
        mom1 = self.glm_model.score_factor(params[:-k_slack])[:, None] * self.instrument
        if k_slack > 0:
            mom1[:, self.slack_idx] += params[-k_slack:]
        return mom1


# In[31]:

mod_red3 = GMMGLMVADD(data_endog, exog_reduced, data_exog, glm_model=glm_binom2, slack_idx=[2])
res_gmm_red3 = mod_red3.fit(np.concatenate((res2.params*0.5, [0])), maxiter=2)
xnames = ['const', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'slack2']
print(res_gmm_red3.summary(xname=xnames))


# In[32]:

mod_red3.slack_idx, mod_red3.k_slack


# In[33]:

res_gmm_red3.bse


# In[34]:

res_gmm_red3.pvalues


# In[35]:

mod_red3.data.xnames


# In[36]:

res2.params


# In[37]:

res2.bse


# In[38]:

res2.pvalues


# In[39]:

scs, scpvalue, _ = res2.model.score_test(res2.params, exog_extra = data_exog[:, 2])


# In[40]:

res_gmm_red3.pvalues[-1], scpvalue, res_gmm_red3.pvalues[-1] - scpvalue, res_gmm_red3.pvalues[-1] / scpvalue - 1


# In[41]:

scs, res_gmm_red3.tvalues[-1]**2, scs / res_gmm_red3.tvalues[-1]**2 - 1


# In[42]:

res2.model.score_test(res2.params, exog_extra = data_exog[:, 2])


#### Explicit Calculation of score tests

# Finally, we can calculate the score or Lagrange Multiplier tests directly based on the matrices that can be calculated with the models.
# 
# As starting point we use the full model with all moment conditions and the parameter estimates of the restricted model.

# In[43]:

res_gmm_red3.params[:-1], res2.params


# In[44]:

params_restricted = np.insert(res_gmm_red3.params[:-1], 2, 0)
params_restricted = np.insert(res2.params, 2, 0)
params_restricted


# In[45]:

M = mod.momcond_mean(params_restricted)
D = mod.gradient_momcond(params_restricted)
moms = mod.momcond(params_restricted)
#V = mod.calc_weightmatrix(...)   check this
V = np.cov(moms.T)
W = res_gmm.weights
M, D, W, V


# In[46]:

from scipy import stats
nobs = mod.endog.shape[0]
Vinv = np.linalg.pinv(V)
ddf = len(res2.params)
ss = (nobs - ddf) * M.dot(Vinv.dot(M))
ss, stats.chi2.sf(ss, 1)


# In[47]:

M


# In[48]:

nobs


# In[49]:

scs, scpvalue


# In[50]:

M.dot(V.dot(M))


# In[51]:

Vb = D.dot(Vinv).dot(D)
Vbinv = np.linalg.pinv(V)
ss = (nobs - ddf) * M.dot(Vbinv.dot(M))
ss, stats.chi2.sf(ss, 1)


# In[52]:

np.sqrt(np.diag(Vbinv))


# In[53]:

res2.bse


# In[54]:

23.94470765 / 0.27311515


# In[55]:

np.sqrt(nobs)


# In[56]:

2.67108656 / 0.03204911


# In[57]:

Vb = D.dot(Vinv).dot(D)
Vbinv = np.linalg.pinv(V)
ss = (nobs - 0) * M.dot(Vbinv.dot(M))
ss, stats.chi2.sf(ss, 1)


# In[58]:

k_params = len(params_restricted)
R = np.zeros((1, k_params))
R[0, 2] = 1
R.dot(params_restricted)


# In[59]:

ss = (nobs - 0) * M[[2]].dot(R.dot(Vbinv.dot(R.T)).dot(M[[2]]))
ss, stats.chi2.sf(ss, 1)


# In[60]:

M[[2]]


# In[60]:



