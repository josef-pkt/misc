
# coding: utf-8

# # One proportion: Hypothesis Tests, Sample Size and Power

# This is a experimental notebook to try to organize various parts for hypothesis tests and related methods.
# 
# This should roughly include the following
# 
# - given a sample
#   - estimate parameter or effect size
#   - hypothesis test given Null and Alternative
#   - confidence interval
# 
# - prospective or evaluative
#   - size of test and power
#   - sample size required
# 
# - sensitivity to misspecification
#   - bias of estimate and of estimated variance
#   - size and power of hypothesis tests
# 

# In[12]:

from __future__ import division   # for py2 compatibility, I'm using Python 3.4
import numpy as np
from scipy import stats
import statsmodels.stats.proportion as smprop
import statsmodels.stats.power as smpow

import pandas as pd # to store results with labels


# ## Sample

# In[13]:

p_true = 0.3
nobs = 30
p_null = p_true

#y = np.random.binomial(nobs, p_true)
y = 7


# Assume we have observed 7 events in a sample of size 30. What are our estimates, confidence interval, and test whether the true proportion = 0.3.

# In[15]:

count = y  # alias
prop_mle = count / nobs
confint_methods = ['beta', 'wilson', 'normal', 'agresti_coull', 'jeffrey']
confints = [smprop.proportion_confint(count, nobs, alpha=0.05, method=method) for method in confint_methods]
ci_df = pd.DataFrame(confints, index=confint_methods, columns=['lower', 'upper'])
print('estimate: ', prop_mle)
ci_df


# **Two sided hypothesis**

# In[16]:

smprop.binom_test(count, nobs, prop=p_null, alternative='two-sided')


# In[19]:

smprop.proportions_ztest(count, nobs, value=p_null, alternative='two-sided')


# In[20]:

smprop.proportions_ztest(count, nobs, value=p_null, alternative='two-sided', prop_var=p_null)


# **Equivalence**

# In[28]:

low, upp = ci_df.loc['beta', :]
smprop.binom_tost(count, nobs, low, upp)


# In[36]:

print('score', smprop.binom_tost(count, nobs, *ci_df.loc['wilson', :]))
print('wald ', smprop.binom_tost(count, nobs, *ci_df.loc['normal', :]))


# In[32]:

smprop.proportions_ztost(count, nobs, *ci_df.loc['wilson', :])


# In[33]:

smprop.proportions_ztost(count, nobs, *ci_df.loc['beta', :])


# **One-sided tests**
# 
# 
# The null nypothesis and alternative hypothesis for alternative `'larger'` are
# 
# H0: p = p0    
# H1: p > p0
# 
# where p0 = 0.3

# In[50]:

te = smprop.binom_test(count, nobs, prop=p_null, alternative='larger')
tw = smprop.proportions_ztest(count, nobs, value=p_null, alternative='larger')
ts = smprop.proportions_ztest(count, nobs, value=p_null, alternative='larger', prop_var=p_null)
print('exact: ', te)
print('wald:  ', tw[1])
print('score: ', ts[1])


# The null nypothesis and alternative hypothesis for alternative `'smaller'` are
# 
# H0: p = p0  
# H1: p < p0
# 
# where p0 = 0.3

# In[51]:

te = smprop.binom_test(count, nobs, prop=p_null, alternative='smaller')
tw = smprop.proportions_ztest(count, nobs, value=p_null, alternative='smaller')
ts = smprop.proportions_ztest(count, nobs, value=p_null, alternative='smaller', prop_var=p_null)
print('exact: ', te)
print('wald:  ', tw[1])
print('score: ', ts[1])


# We can look at null hypothesis that are further away from the observed proportion to see which hypothesis are rejected. The observed proportion is 0.23, our new null hypothesis value is 0.6. 

# In[67]:

p_null_ = 0.6
te = smprop.binom_test(count, nobs, prop=p_null_, alternative='smaller')
tw = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='smaller')
ts = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='smaller', prop_var=p_null)
print('exact: ', te)
print('wald:  ', tw[1])
print('score: ', ts[1])


# In[65]:

p_null_ = 0.6
te = smprop.binom_test(count, nobs, prop=p_null_, alternative='larger')
tw = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='larger')
ts = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='larger', prop_var=p_null)
print('exact: ', te)
print('wald:  ', tw[1])
print('score: ', ts[1])


# The `smaller` hypothesis is strongly rejected, which means that we reject the null hypothesis that the true proportion is 0.6 or larger in favor of the alternative hypothesis that the true proportion is smaller than 0.6.
# 
# In the case `larger` alternative, the p-value is very large and we cannot reject the Null hypothesis that the true proportion is 0.6 (or smaller) in favor of the hypothesis that the true proportion is larger than 0.6. 
# 
# Non-inferiority and superiority tests are special cases of these one-sided tests where the specific case is defined in terms of deviations from a benchmark value. The null hypothesis for a non-inferiority test can be defined, for example, by being less than a specified amount 5% below a benchmark proportion. If we reject the test, then we conclude that the proportion is not worse than 5% below the benchmark, at the given confidence level of the test.

# **Aside: Inequality Null hypothesis**
# 
# For most methods the p-values for the hypothesis tests are the same for the case when the null hypothesis is and inequality 
# 
# 
# The null nypothesis and alternative hypothesis for alternative `'larger'` specify that the true proportion is smaller than or equal to the hypothesized value versus the alternative that it is larger.
# 
# H0': p <= p0    
# H1': p > p0
# 
# 
# The null nypothesis and alternative hypothesis for alternative 'smaller' are
# 
# H0': p >= p0  
# H1': p < p0
# 
# 
# The score test is an exception to this. If the null hypothesis is a inequality, then the constrained maximum likelihood estimate will depend on whether the constraint of the null hypothesis is binding or not. If it is binding, then the score test is the same as for the test with an equality in the null hypothesis. If the constrained is not binding then the null parameter estimate is the same as the estimate used for the Wald test.
# Because the equality is the worst case in these hypothesis test, it does not affect the validity of the tests. However, in the asymptotic tests it would add another option to define the variance used in the calculations, and the standard score test does not take the inequality into account in calculating the variance. This is not implemented, so we restrict ourselves to equality null hypothesis, even though the interpretation is mostly the same as for the inequality null hypothesis.
# 

# In[ ]:




# **Standard t-test**
# 
# We can also use the standard t-test in large samples if we encode the data with 0 for no event and 1 for the success event. The t-test estimates the variance from the data and does not take the relationship between mean and variance explicitly into account. However, by the law of large numbers the mean, i.e. the proportion in the current case, will be asymptotically distributed as normal which can be approximated by the t-distribution.

# In[61]:

import statsmodels.stats.weightstats as smsw
yy = np.repeat([0, 1], [nobs - count, count])
ds = smsw.DescrStatsW(yy)
ds.ttest_mean(0.3)


# In[62]:

vars(ds)


# In[63]:

ds.ttest_mean(0.3, alternative='larger')


# In[64]:

ds.ttest_mean(0.3, alternative='smaller')


# In this example the p-values from the t-test are in between the asymptotic score and wald tests based on the normal distribution for all three alternatives. The t-test based toast has a p-value that is slightly larger than the normal distribution based TOST test for proportions, 0.049 versus 0.041 which are both larger than the binomial distribution based TOST, at the latter confidence interval. 

# In[68]:

ds.ttost_mean(*ci_df.loc['beta', :])


# In[ ]:




# In[ ]:




# ## Sample Size and Power

# First we illustrate the rejection region of a test which is the set of all observations at which we reject the null hypothesis.
# Size of a test is the probability to sample an observation in the rejection region under the null hypothesis, power is the probability under the alternative hypothesis.
# 
# The rejection region is a property of the hypothesis test, the following calculates it for the two-sided binomial and the two-sided ztest for a single proportion. This depends on the distribution that we use in the hypothesis test, exact distribution which is the binomial in this case or a normal or t distribution as large sample approximation.
# Ones we have the rejection region, we can also use different distributions for evaluating the power either based on the exact distribution or an a large sample approximation or asymptotic distribution.
# 
# The sample size that is required to achieve at least a desired power under a given alternative can be explicitly calculated in the special case one-sided tests where both the hypothesis test distribution and the distribution for the power calculations are the normal distribution. In almost all other cases we have to use an iterative solver to find the required sample size.
# 
# Power and sample size calculation are currently only implemented for one approximation and for equivalence tests. In the following we illustrate several methods for calculating the power which will be useful for different cases depending on whether simplification or computational shortcuts exist or not.
# 

# **Rejection region**

# In[103]:

rej = np.array([smprop.proportions_ztest(count_, nobs, value=p_null, alternative='two-sided', prop_var=p_null)[1] 
                for count_ in range(nobs + 1)])
rej_indicator = (rej < 0.05) #.astype(int)
np.column_stack((rej, rej_indicator))
rej_indicator_score = rej_indicator  # keep for later use


# In[104]:

rej = np.array([smprop.binom_test(count_, nobs, prop=p_null, alternative='two-sided') for count_ in range(nobs + 1)])
rej_indicator = (rej < 0.05) #.astype(int)
np.column_stack((range(nobs + 1), rej, rej_indicator))


# ### Power calculation - a general method

# In a general method we can use the rejection region of a hypothesis test directly to calculate the probability.
# 
# We can use the set of values for which the null hypothesis is rejected instead of using a boolean indicator.

# In[105]:

x = np.arange(nobs + 1)
x_rej = x[rej_indicator]
x_rej_score = x[rej_indicator_score]


# In[113]:

print('binom', x_rej)
print('score', x_rej_score)


# The rejection region of the score test is larger than the one of the exact binomial test. The score test rejects also if 14 events are observed.

# For the current case we use the exact binomial distribution to calculate the power. The null hypothesis in this example is a two-sided test for p = 0.3. Use p1 for the proportion at which the power or rejection probability is calculated. First we check the size of the test, i.e. p1 = p_null = 0.3

# In[114]:

p1 = 0.3
stats.binom.pmf(x_rej, nobs, p1).sum()


# Because we are using the exact test, the probability of rejection under the null is smaller than the required alpha = 0.05. In this example the exact probability is close to the 0.05 threshold. In contrast to this, the score test is liberal in this example and rejects with probability 0.07 instead of the required 0.05.

# In[112]:

stats.binom.pmf(x_rej_score, nobs, p1).sum()


# This method with explicit enumeration of the rejection values can be used for any distribution but will require more computational time than explicit calculations that take advantage of the specific structure. In the case of one sample or one parameter hypothesis test, the rejection region consist of two tail intervals. If we have the boundary of the rejection region available, then we can directly use the cumulative distribution or the survival function to calculate the tail probabilities.
# 
# In the case of the binomial distribution with probability p_null under the null hypothesis has tail probabilities at most alpha / 2 in each tail (for equal tailed hypothesis tests).

# In[129]:

lowi, uppi = stats.binom.interval(0.95, nobs, p_null)
lowi, uppi


# **Detour: open or close interval**
# 
# The cdf is defined by a weak inequality cdf(t) = Prob(x <= t), the survival function sf is defined by a strict inequality sf(t) = Prob(x > t) so that cdf(t) + sf(t) = 1. Whether the inequalities are strict or weak does not make a difference in continuous distributions that don't have mass points. However, it does make a difference for discrete distribution. If we want a tail probability of alpha, then the cdf has this tail probability including the boundary point, while the sf excludes the boundary point. So to have a upper tail probability alpha for a t such that Prob(x >= t) < alpha but close to it, we need to use sf(t - 1). similarly, we have to subtract one if we want an open interval for the cdf at the lower tail.
# 
# Specifically, define the lower and upper thresholds that are in the rejection region
# 
# low = max{x: prob(x <= t) <= alpha / 2 
# upp = min{x: prob(x >= t) <= alpha / 2 
# 
# Because of the discreteness of the sample space having tail probabilities equal to alpha / 2 is in general not possible.
# 

# In[130]:

low, upp = lowi, uppi


# In[132]:

stats.binom.ppf(0.025, nobs, p_null), stats.binom.isf(0.025, nobs, p_null)


# If we reject at 4 and smaller and reject at 14 and larger, then the probability of rejection is larger than 0.025 in each tail:

# In[133]:

stats.binom.cdf(low, nobs, p_null), stats.binom.sf(upp - 1, nobs, p_null)


# If we shrink the rejection region in each tail by one, so we reject at 3 and smaller and reject at 15 and larger, then the probability of rejection is smaller than 0.025 in each tail. The total rejection probability is at 0.026 smaller than 0.05 and shows the typical case that exact tests are conservative, i.e. reject less often than alpha, often considerably less:

# In[137]:

prob_low = stats.binom.cdf(low - 1, nobs, p_null)
prob_upp = stats.binom.sf(upp, nobs, p_null)
prob_low, prob_upp, prob_low + prob_upp


# In this case we can increase the lower rejection threshold by one and still stay below the total rejection probability of 0.05, although in this case the rejection probability in the lower tail is larger than 0.025. In this example the same also works on the other side by expanding only the rejection region in the upper tail.

# In[138]:

prob_low = stats.binom.cdf(low, nobs, p_null)
prob_upp = stats.binom.sf(upp, nobs, p_null)
prob_low, prob_upp, prob_low + prob_upp


# In[139]:

prob_low = stats.binom.cdf(low - 1, nobs, p_null)
prob_upp = stats.binom.sf(upp - 1, nobs, p_null)
prob_low, prob_upp, prob_low + prob_upp


# In[124]:

stats.binom.cdf(upp, nobs, p_null) - stats.binom.cdf(low, nobs, p_null)


# TODO: why does binom_test reject at 4? 
# binom_test is used from scipy.stats for the two-sided alternative.

# In[142]:

smprop.binom_test(3, nobs, prop=p_null, alternative='smaller'), smprop.binom_test(4, nobs, prop=p_null, alternative='smaller')


# In[144]:

smprop.binom_test(4, nobs, prop=p_null, alternative='two-sided')
# we get the same answer as in R
# in R binom.test(4,30, 0.3, alternative="two.sided")  --> 0.04709225


# The binomial test is not a centered test. It looks like it adds the probability from the further away tail for all x that have lower pmf than the observed value.
# check with Fay for the three different ways of defining two-tailed tests and confints.
# 
# The pvalue for the centered test is based on doubling the probability of the smaller tail. Given that it does not exist, we can implement it quickly, and check against R's exactci package, which matches our results.

# In[151]:

def binom_test_centered(count, nobs, prop=0.5):
    """two-sided centered binomial test"""
    prob_low = stats.binom.cdf(count, nobs, p_null)
    prob_upp = stats.binom.sf(count - 1, nobs, p_null)
    return 2 * min(prob_low, prob_upp)


# In[152]:

binom_test_centered(3, nobs, prop=p_null), binom_test_centered(4, nobs, prop=p_null)


# In[153]:

binom_test_centered(13, nobs, prop=p_null), binom_test_centered(14, nobs, prop=p_null)

results from R library exactci, with centered binomial test

> be = binom.exact(3, 30, p = 0.3)
> be$p.value
[1] 0.01863313
> be = binom.exact(4, 30, p = 0.3)
> be$p.value
[1] 0.06030989
> be = binom.exact(13, 30, p = 0.3)
> be$p.value
[1] 0.1689401
> be = binom.exact(14, 30, p = 0.3)
> be$p.value
[1] 0.0801051
# ## Back to power

# After this more extended detour we go back to our power calculations. So assuming we know the critical values of our rejection region, we can calculate the power using the cdf and sf function of the binomial distribution.

# In[155]:

def power_binom_reject(low, upp, prop, nobs):
    """ calculate the power of a test given the rejection intervals
    
    This assumes that the rejection region is the union of the lower 
    tail up to and including low, and the upper tail starting at and
    including upp.
    
    The Binomial distribution is used to calculate the power.
    
    Parameters
    ----------
    low
    upp
    prop : float in interval (0, 1)
        proportion parameter for the binomial distribution
    nobs : int
        number of trials for binomial distribution
        
    Returns
    -------
    power : float
        Probability of rejection if the true proportion is `prop`.
        
    Notes
    -----
    Works in vectorized form with appropriate arguments, i.e. 
    nonscalar arguments are numpy arrays that broadcast correctly.
    
    """
    prob_low = prob_upp = 0   # initialize
    if low is not None:
        prob_low = stats.binom.cdf(low, nobs, prop)
    if upp is not None:
        prob_upp = stats.binom.sf(upp - 1, nobs, prop)
    return prob_low + prob_upp


# In[162]:

for test, l, u in [('binom        ', 4, 15), ('binom_central', 3, 15), ('score        ', 4, 14)]:
    print(test, l, u, power_binom_reject(l, u, p_null, nobs))


# In[ ]:




# In[ ]:




# ## trying more
# 
# The rest below is just some unsorted experiments to try a few more things.

# In[ ]:




# TODO: The following is not correct because when we change the sample size, then the rejection region also changes.

# In[164]:

[power_binom_reject(4, 15, p_null, nobs_) for nobs_ in range(30, 50)]


# We can also calculate this in vectorized form for the set of sample sizes and all three tests:

# In[166]:

power_binom_reject(np.array([4, 3, 4]), np.array([15, 15, 14]), p_null, np.arange(30, 50)[:, None])


# In[ ]:




# In[ ]:




# ## Trying out two sample proportion, incorrect if nobs is scalar instead of same length as count.

# In[81]:

smprop.proportions_ztest(np.array([6,7]), nobs, value=0, alternative='two-sided', prop_var=p_null)


# In[77]:

smprop.proportions_ztest(np.array([6,7]), nobs*np.ones(2), value=1/30, alternative='two-sided', prop_var=p_null)


# In[78]:

smprop.proportions_ztest(np.array([6,7]), nobs, value=1/30, alternative='two-sided', prop_var=p_null)


# In[79]:

smprop.proportions_ztest(np.array([6,7]), nobs, value=-1/30, alternative='two-sided', prop_var=p_null)


# In[80]:

smprop.proportions_ztest(np.array([6,7]), nobs*np.ones(2), value=-1/30, alternative='two-sided', prop_var=p_null)


# In[ ]:




# In[4]:

get_ipython().magic('pinfo smprop.proportion_confint')


# In[ ]:

smprop.proportion_confint()


# In[11]:

from statsmodels.stats.proportion import proportion_effectsize
es = proportion_effectsize(0.4, 0.5)
smpow.NormalIndPower().solve_power(es, nobs1=60, alpha=0.05, ratio=0)
# R pwr 0.3447014091272153


# In[14]:

smpow.NormalIndPower().solve_power(proportion_effectsize(0.4, 0.5), nobs1=None, alpha=0.05, ratio=0, power=0.9)


# In[ ]:




# In[ ]:




# In[ ]:




# In[25]:

low, upp, nobs, p_alt = 0.7, 0.9, 509/2, 0.82
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.025, dist='norm',
                     variance_prop=None, discrete=True, continuity=0,
                     critval_continuity=0)
    


# In[39]:

low, upp, nobs, p_alt = 0.7, 0.9, 419/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm',
                     variance_prop=None, discrete=False, continuity=0,
                     critval_continuity=0)


# In[41]:

low, upp, nobs, p_alt = 0.7, 0.9, 417/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm',
                     variance_prop=None, discrete=False, continuity=1,
                     critval_continuity=0)


# In[49]:

low, upp, nobs, p_alt = 0.7, 0.9, 420/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='binom',
                     variance_prop=None, discrete=False, continuity=0,
                     critval_continuity=0)


# In[55]:

low, upp, nobs, p_alt = 0.7, 0.9, 414/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.025, dist='norm',
                     variance_prop=None, discrete=False, continuity=1,
                     critval_continuity=0)


# In[ ]:




# In[ ]:




# In[71]:

low, upp, nobs = 0.4, 0.6, 100
smprop.binom_tost_reject_interval(low, upp, nobs, alpha=0.05)


# In[59]:

value, nobs = 0.4, 50
smprop.binom_test_reject_interval(value, nobs, alpha=0.05)


# In[70]:

smprop.proportion_confint(50, 100, method='beta')


# In[72]:

low, upp, nobs = 0.7, 0.9, 100
smprop.binom_tost_reject_interval(low, upp, nobs, alpha=0.05)


# In[76]:

low, upp, nobs, p_alt = 0.7, 0.9, 100, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='binom',
                     variance_prop=None, discrete=False, continuity=0,
                     critval_continuity=0)


# In[78]:

low, upp, nobs, p_alt = 0.7, 0.9, 100, 0.8
smprop.power_binom_tost(low, upp, nobs, p_alt, alpha=0.05)


# In[79]:

low, upp, nobs, p_alt = 0.7, 0.9, 125, 0.8
smprop.power_binom_tost(low, upp, nobs, p_alt, alpha=0.05)


# In[ ]:




# In[ ]:




# In[ ]:




# In[132]:

# from Lachine 1981 equ (3) and (4)

from scipy import stats
def sample_size_normal_greater(diff, std_null, std_alt, alpha=0.05, power=0.9):
    crit_alpha, crit_pow = stats.norm.isf(alpha), stats.norm.isf(1 - power)
    return ((crit_alpha * std_null + crit_pow * std_alt) / np.abs(diff))**2

def power_normal_greater(diff, std_null, std_alt, nobs, alpha=0.05):
    crit_alpha = stats.norm.isf(alpha)
    crit_pow = (np.sqrt(nobs) * np.abs(diff) - crit_alpha * std_null) / std_alt
    return stats.norm.cdf(crit_pow)


# In[140]:

# Note for two sample comparison we have to adjust the standard deviation for unequal sample sizes
n_frac1 = 0.5
n_frac2 = 1 - frac1

# if defined by ratio: n2 = ratio * n1
ratio = 1
n_frac1 = 1 / ( 1. + ratio)
n_frac2 = 1 - frac1


# If we use fraction of nobs, then sample_size return nobs is total number of observations
diff = 0.2
std_null = std_alt = 1 * np.sqrt(1 / 0.5 + 1 / 0.5)
nobs = sample_size_normal_greater(diff, std_null, std_alt, alpha=0.05, power=0.9)
nobs


# In[134]:

#nobs = 858
power_normal_greater(diff, std_null, std_alt, nobs, alpha=0.05)


# In[135]:

alpha=0.05; power=0.9
stats.norm.isf(alpha), stats.norm.isf(1 - power)


# In[136]:

crit_alpha = stats.norm.isf(alpha)
(np.sqrt(nobs) * np.abs(diff) - crit_alpha * std_null) / std_alt


# In[137]:

stats.norm.cdf(_)


# In[138]:

smprop.binom_test_reject_interval([0.4, 0.6], [100], alpha=0.05)


# In[ ]:



