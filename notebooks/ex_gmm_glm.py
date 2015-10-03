
# coding: utf-8

## GLM as a GMM Model

# The purpose of this notebook is to replicate generalized linear models, or generalized estimating equations with a independence correlation structure, as a Generalized Methods of Moments model.
# 
# The second part compares a score test for variable addition with a test of overidentifying restriction which should eventually provide a generic framework of conditional moment tests. 
# 
# **Status**: We can replicate GLM by GMM, but the rest is experimental

# In[29]:

from __future__ import division
import numpy as np
from statsmodels.sandbox.regression.gmm import GMM


# In[30]:

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

# In[ ]:

import statsmodels.api as sm


# In[ ]:

# stardata is binomial count we 2d endog, Doesn't work yet.

data = sm.datasets.star98.load()
data_exog = sm.add_constant(data.exog, prepend=True)[:, :8]


# In[80]:

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


# In[49]:

res_binom_hc0 = glm_binom.fit(cov_type='HC0')
print(res_binom_hc0.bse)


# In[60]:

mod = GMMGLM(data_endog, data_exog, glm_model=glm_binom)
res_gmm = mod.fit(res.params*0.5, maxiter=1)#, optim_method='nm')


# In[61]:

print(res_gmm.params)


# In[62]:

print(res_gmm.summary())


# In[63]:

mod.momcond(res.params).mean(0)


# In[64]:

res.model.score(res.params)


# GMM produces the same result as GLM-Logit with robust standard errors

# In[ ]:

res_gmm.bse / res_binom_hc0.bse - 1


# In[67]:

res_gmm.bse - res_binom_hc0.bse


# In[ ]:




### Application: score test for added variable

# This is experimental and might not work yet.
# 
# The third variable `educ` has a p-value of 0.012 in the wald t_test based on the asymptotic normal distribution using the heteroscedasticity robust standard errors. Since wald test are in many cases liberal, we can compare this with a score test which has in many cases more accurate p-values. For simplicity we just want to test the null hypothesis that the effect of education is zero.
# 
# In the following we drop the variable from the estimated model and calculate score and conditional moment tests. In the GMM version we estimate the reduced model with `educ` as additional instrument, and look at the test for overidentifying restrictions.
# 
# The following will be easier to do with formulas and pandas DataFrames, but I want to minimize possible sources of errors, and I'm more comfortable with numpy.

# In[70]:

affair_mod.model.data.xnames


# In[73]:

idx = list(range(len(res.params)))
del idx[2]
print(res.params[idx])  # check that we don't have -0.0392 at index 2 anymore


# In[75]:

exog_reduced = data_exog[:, idx]   # exog without educ


# In[84]:

glm_binom2 = sm.GLM(data_endog, exog_reduced, family=sm.families.Binomial())
res2 = glm_binom2.fit()


# In[85]:

res2.model.score_test(res2.params, exog_extra = data_exog[:, 2])


# The pvalue of the score test, 0.01123, is close to the pvalue of the robust wald test and almost identical to the pvalue of the nonrobust wald test.

# In[86]:

res.pvalues[2]


# Next we try the test for overidentifying restriction in GMM

# In[88]:

mod_red = GMMGLM(data_endog, exog_reduced, data_exog, glm_model=glm_binom2)
res_gmm_red = mod_red.fit(res2.params*0.5, maxiter=1)
print(res_gmm_red.summary())


# In[89]:

res_gmm_red.jtest()


# The overidentifying restrictions are not rejected. 
# Note: This is a bit of a weird test in this case. We have the reduced model in the prediction, but the extra variable as an instrument. 
#     I'm not sure what we are actually testing, besides some general orthogonality condition.
#     
# What we would like to do right now is to use the restricted parameters, so we can calculate the moment conditions in the constrained model. I don't see yet where this is supported or how to support it. 
# 
# The following are just some additional checks with respect to changes in the number of GMM iterations. The above uses the identity weight matrix which is irrelevant in the exactly identified case that we had before. Now, we are in the overidentified case and need to adjust the weight matrix. As it turns out below, using two or more GMM iterations reduces the p-value for the j-test to 0.011 which is almost exactly the same as the non-robust Wald and the score test. (I need to think.)

# In[90]:

res2.params


# In[91]:

res_gmm_red0 = mod_red.fit(res2.params*0.5, maxiter=0)
res_gmm_red2 = mod_red.fit(res2.params*0.5, maxiter=2)
print(res_gmm_red0.params)
print(res_gmm_red2.params)


# In[94]:

res2.params*0.5  # check how much params moved with maxiter=0


# In[95]:

res_gmm_red2.jtest()


# In[ ]:




# In[98]:

res_gmm_red10 = mod_red.fit(res2.params*0.5, maxiter=10, optim_args=dict(disp=0)) # turn of the display of iteration results
res_gmm_red10.jtest()


# In[99]:

res_gmm_red10.params


# In[100]:

res2.params


# This looks like the expected results after all. After 10 GMM iterations the parameter estimates get close to the constrained GLM estimates.
# 
# We are getting closer but this still solves a different problem since we don't keep the parameters fixed at the constrained solution.
# 
# More to come: In another branch I have generic standalone conditional moment tests, that are similar to the GMM setup but use fixed parameters estimated in the constrained model.

# Using another option, we can get the constrained parameter estimate by assigning zero weight to the extra restriction. The parameter estimates are close to the constrained GLM estimates. 
# But I never tried this before and I'm not sure whether the weights are inverse weights in this example. Also, since the estimation is not based on the efficient weights matrix, the jtest does not apply, even though the result is still available.

# In[108]:

inv_weights = np.eye(9)
inv_weights[2,2] = 0
res_gmm_red1w = mod_red.fit(res2.params*0.5, maxiter=1, inv_weights=inv_weights, optim_args=dict(disp=0), has_optimal_weights=False)
print(res_gmm_red1w.params)
print(res_gmm_red1w.jtest())


# **Conclusion** so far is that I don't know yet how to get the equivalent of an score test for a constrained model with GMM
