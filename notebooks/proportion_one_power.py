
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

# In[1]:

from __future__ import division   # for py2 compatibility, I'm using Python 3.4
import numpy as np
from scipy import stats
import statsmodels.stats.proportion as smprop
import statsmodels.stats.power as smpow

import pandas as pd # to store results with labels


# In[2]:

get_ipython().magic('matplotlib inline')


# ## Sample

# In[3]:

p_true = 0.3
nobs = 30
p_null = p_true

#y = np.random.binomial(nobs, p_true)
y = 7


# Assume we have observed 7 events in a sample of size 30. What are our estimates, confidence interval, and test whether the true proportion = 0.3.

# In[4]:

count = y  # alias
prop_mle = count / nobs
confint_methods = ['beta', 'wilson', 'normal', 'agresti_coull', 'jeffrey']
confints = [smprop.proportion_confint(count, nobs, alpha=0.05, method=method) for method in confint_methods]
ci_df = pd.DataFrame(confints, index=confint_methods, columns=['lower', 'upper'])
print('estimate: ', prop_mle)
ci_df


# We can check some corner case behavior to see if the function handles those correctly. It does not yet do so. beta/exact confidence interval contains a NaN if the count is all of the same kind, normal and agresti_coull return proportions that are negative or larger than one. (I opened https://github.com/statsmodels/statsmodels/issues/2742 )

# In[5]:

count_ = 0
confints0 = [smprop.proportion_confint(count_, nobs, alpha=0.05, method=method) for method in confint_methods]
count_ = 1
confints1 = [smprop.proportion_confint(count_, nobs, alpha=0.05, method=method) for method in confint_methods]
count_ = nobs - 1
confintsnm1 = [smprop.proportion_confint(count_, nobs, alpha=0.05, method=method) for method in confint_methods]
count_ = nobs
confintsn = [smprop.proportion_confint(count_, nobs, alpha=0.05, method=method) for method in confint_methods]
pd.DataFrame(np.column_stack((confints0, confints1, confintsnm1, confintsn)), index=confint_methods, 
             columns=['0 lower', '0 upper', '1 lower', '1 upper', 'n-1 lower', 'n-1 upper', 'n lower', 'n upper'])


# **Two sided hypothesis**

# In[6]:

smprop.binom_test(count, nobs, prop=p_null, alternative='two-sided')


# In[7]:

smprop.proportions_ztest(count, nobs, value=p_null, alternative='two-sided')


# In[8]:

smprop.proportions_ztest(count, nobs, value=p_null, alternative='two-sided', prop_var=p_null)


# **Aside: Corner case for tests**
# 
# Many normal distribution based hypothesis tests have problems with observations where the count is zero. Various solutions have been proposed, one of them is to add 0.5 to all zero observations. PASS also adds a small number like 0.001 for the power calculations in this case. There is currently no option for this in my functions.

# In[9]:

print('x = 0')
count_ = 0
p_null_ = 0.05
print(smprop.binom_test(count_, nobs, prop=p_null_, alternative='two-sided'))
print(smprop.proportions_ztest(count_, nobs, value=p_null_, alternative='two-sided'))
print(smprop.proportions_ztest(count_, nobs, value=p_null_, alternative='two-sided', prop_var=p_null_))
print('\nx = 1')
count_ = 0.05
p_null_ = 0.05
print(smprop.binom_test(count_, nobs, prop=p_null_, alternative='two-sided'))
print(smprop.proportions_ztest(count_, nobs, value=p_null_, alternative='two-sided'))
print(smprop.proportions_ztest(count_, nobs, value=p_null_, alternative='two-sided', prop_var=p_null_))


# In[10]:

import statsmodels.api as sm

res = sm.GLM([[7, 30 - 7]], [[1]], family=sm.genmod.families.Binomial(link=sm.genmod.families.links.identity)).fit()
print(res.summary())


# In[11]:

tt = res.t_test('const - %f' % p_null)
print(tt)
'HO: const = %f' % p_null, tt.pvalue, tt.conf_int()


# The pvalue is exactly the same as the Wald test version of `proportions_ztest`. The confidence interval is identical to `proportion_confint` with method `"normal"`.

# In[12]:

res = sm.GLM([[7, 30 - 7]], [[1]], family=sm.genmod.families.Binomial(link=sm.genmod.families.links.identity), offset=[[p_null]]).fit()
print(res.summary())


# **Equivalence**

# In[13]:

low, upp = ci_df.loc['beta', :]
smprop.binom_tost(count, nobs, low, upp)


# In[14]:

print('score', smprop.binom_tost(count, nobs, *ci_df.loc['wilson', :]))
print('wald ', smprop.binom_tost(count, nobs, *ci_df.loc['normal', :]))


# In[15]:

smprop.proportions_ztost(count, nobs, *ci_df.loc['wilson', :])


# In[16]:

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

# In[17]:

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

# In[18]:

te = smprop.binom_test(count, nobs, prop=p_null, alternative='smaller')
tw = smprop.proportions_ztest(count, nobs, value=p_null, alternative='smaller')
ts = smprop.proportions_ztest(count, nobs, value=p_null, alternative='smaller', prop_var=p_null)
print('exact: ', te)
print('wald:  ', tw[1])
print('score: ', ts[1])


# We can look at null hypothesis that are further away from the observed proportion to see which hypothesis are rejected. The observed proportion is 0.23, our new null hypothesis value is 0.6. 

# In[19]:

p_null_ = 0.6
te = smprop.binom_test(count, nobs, prop=p_null_, alternative='smaller')
tw = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='smaller')
ts = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='smaller', prop_var=p_null)
print('exact: ', te)
print('wald:  ', tw[1])
print('score: ', ts[1])


# In[20]:

p_null_ = 0.6
te = smprop.binom_test(count, nobs, prop=p_null_, alternative='larger')
tw = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='larger')
ts = smprop.proportions_ztest(count, nobs, value=p_null_, alternative='larger', prop_var=p_null)
print('exact: ', te)
print('wald:  ', tw[1])
print('score: ', ts[1])


# The `smaller` hypothesis is strongly rejected, which means that we reject the null hypothesis that the true proportion is 0.6 or larger in favor of the alternative hypothesis that the true proportion is smaller than 0.6.
# 
# In the case of the `larger` alternative, the p-value is very large and we cannot reject the null hypothesis that the true proportion is 0.6 (or smaller) in favor of the hypothesis that the true proportion is larger than 0.6. 
# 
# Non-inferiority and superiority tests are special cases of these one-sided tests. Often, the specific case is defined in terms of deviations from a benchmark value. The null hypothesis for a non-inferiority test can be defined, for example, by being less than a specified amount, say 5%, below a benchmark proportion. If we reject the test, then we conclude that the proportion is not worse than 5% below the benchmark, at the given confidence level of the test.

# **Aside: Inequality Null hypothesis**
# 
# In the above definition of the null hypothesis we used an equality. For most methods the p-values for the hypothesis tests are the same for the case when the null hypothesis is an inequality 
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
# Reference for a score analysis with inequality null hypothesis for the case of comparing two proportions, see ...
# 

# In[ ]:




# **Standard t-test**
# 
# We can also use the standard t-test in large samples if we encode the data with 0 for no event and 1 for the success event. The t-test estimates the variance from the data and does not take the relationship between mean and variance explicitly into account. However, by the law of large numbers the mean, i.e. the proportion in the current case, will be asymptotically distributed as normal which can be approximated by the t-distribution.

# In[21]:

import statsmodels.stats.weightstats as smsw
yy = np.repeat([0, 1], [nobs - count, count])
ds = smsw.DescrStatsW(yy)
ds.ttest_mean(0.3)


# In[22]:

vars(ds)


# In[23]:

ds.ttest_mean(0.3, alternative='larger')


# In[24]:

ds.ttest_mean(0.3, alternative='smaller')


# In this example the p-values from the t-test are in between the asymptotic score and wald tests based on the normal distribution for all three alternatives. The t-test based toast has a p-value that is slightly larger than the normal distribution based TOST test for proportions, 0.049 versus 0.041 which are both larger than the binomial distribution based TOST, which is 0.025 when we use the latter's confidence interval for the equivalence margins. 

# In[25]:

ds.ttost_mean(*ci_df.loc['beta', :])


# We used a full sample with individual observations in the above. However, `DescrStatsW` allows us to use weights and we can specify the sample by the frequency of each level of the observation. The results are the same as before.

# In[26]:

ds2 = smsw.DescrStatsW([0, 1], weights=[nobs - count, count])
ds2.ttest_mean(0.3, alternative='smaller')


# In[27]:

ds2.ttost_mean(*ci_df.loc['beta', :])


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

# In[28]:

rej = np.array([smprop.proportions_ztest(count_, nobs, value=p_null, alternative='two-sided', prop_var=p_null)[1] 
                for count_ in range(nobs + 1)])
rej_indicator = (rej < 0.05) #.astype(int)
np.column_stack((rej, rej_indicator))
rej_indicator_score = rej_indicator  # keep for later use


# In[29]:

rej = np.array([smprop.binom_test(count_, nobs, prop=p_null, alternative='two-sided') for count_ in range(nobs + 1)])
rej_indicator = (rej < 0.05) #.astype(int)
np.column_stack((range(nobs + 1), rej, rej_indicator))


# ### Power calculation - a general method

# In a general method we can use the rejection region of a hypothesis test directly to calculate the probability.
# 
# We can use the set of values for which the null hypothesis is rejected instead of using a boolean indicator.

# In[30]:

x = np.arange(nobs + 1)
x_rej = x[rej_indicator]
x_rej_score = x[rej_indicator_score]


# In[31]:

print('binom', x_rej)
print('score', x_rej_score)


# The rejection region of the score test is larger than the one of the exact binomial test. The score test rejects also if 14 events are observed.

# For the current case we use the exact binomial distribution to calculate the power. The null hypothesis in this example is a two-sided test for p = 0.3. Use p1 for the proportion at which the power or rejection probability is calculated. First we check the size of the test, i.e. p1 = p_null = 0.3

# In[32]:

p1 = 0.3
stats.binom.pmf(x_rej, nobs, p1).sum()


# Because we are using the exact test, the probability of rejection under the null is smaller than the required alpha = 0.05. In this example the exact probability is close to the 0.05 threshold. In contrast to this, the score test is liberal in this example and rejects with probability 0.07 instead of the required 0.05.

# In[33]:

stats.binom.pmf(x_rej_score, nobs, p1).sum()


# This method with explicit enumeration of the rejection values can be used for any distribution but will require more computational time than explicit calculations that take advantage of the specific structure. In the case of one sample or one parameter hypothesis test, the rejection region consist of two tail intervals. If we have the boundary of the rejection region available, then we can directly use the cumulative distribution or the survival function to calculate the tail probabilities.
# 
# In the case of the binomial distribution with probability p_null under the null hypothesis has tail probabilities at most alpha / 2 in each tail (for equal tailed hypothesis tests).

# In[34]:

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

# In[35]:

low, upp = lowi, uppi


# In[36]:

stats.binom.ppf(0.025, nobs, p_null), stats.binom.isf(0.025, nobs, p_null)


# If we reject at 4 and smaller and reject at 14 and larger, then the probability of rejection is larger than 0.025 in each tail:

# In[37]:

stats.binom.cdf(low, nobs, p_null), stats.binom.sf(upp - 1, nobs, p_null)


# If we shrink the rejection region in each tail by one, so we reject at 3 and smaller and reject at 15 and larger, then the probability of rejection is smaller than 0.025 in each tail. The total rejection probability is at 0.026 smaller than 0.05 and shows the typical case that exact tests are conservative, i.e. reject less often than alpha, often considerably less:

# In[38]:

prob_low = stats.binom.cdf(low - 1, nobs, p_null)
prob_upp = stats.binom.sf(upp, nobs, p_null)
prob_low, prob_upp, prob_low + prob_upp


# In this case we can increase the lower rejection threshold by one and still stay below the total rejection probability of 0.05, although in this case the rejection probability in the lower tail is larger than 0.025. In this example the same also works on the other side by expanding only the rejection region in the upper tail.

# In[39]:

prob_low = stats.binom.cdf(low, nobs, p_null)
prob_upp = stats.binom.sf(upp, nobs, p_null)
prob_low, prob_upp, prob_low + prob_upp


# In[40]:

prob_low = stats.binom.cdf(low - 1, nobs, p_null)
prob_upp = stats.binom.sf(upp - 1, nobs, p_null)
prob_low, prob_upp, prob_low + prob_upp


# In[41]:

stats.binom.cdf(upp, nobs, p_null) - stats.binom.cdf(low, nobs, p_null)


# TODO: why does binom_test reject at 4? 
# binom_test is used from scipy.stats for the two-sided alternative.

# In[42]:

smprop.binom_test(3, nobs, prop=p_null, alternative='smaller'), smprop.binom_test(4, nobs, prop=p_null, alternative='smaller')


# In[43]:

smprop.binom_test(4, nobs, prop=p_null, alternative='two-sided')
# we get the same answer as in R
# in R binom.test(4,30, 0.3, alternative="two.sided")  --> 0.04709225


# The binomial test is not a centered test. It looks like it adds the probability from the further away tail for all x that have lower pmf than the observed value.
# check with Fay for the three different ways of defining two-tailed tests and confints.
# 
# The pvalue for the centered test is based on doubling the probability of the smaller tail. Given that it does not exist, we can implement it quickly, and check against R's exactci package, which matches our results.

# In[44]:

def binom_test_centered(count, nobs, prop=0.5):
    """two-sided centered binomial test"""
    prob_low = stats.binom.cdf(count, nobs, p_null)
    prob_upp = stats.binom.sf(count - 1, nobs, p_null)
    return 2 * min(prob_low, prob_upp)


# In[45]:

binom_test_centered(3, nobs, prop=p_null), binom_test_centered(4, nobs, prop=p_null)


# In[46]:

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
# ## Exact Power

# After this more extended detour we go back to our power calculations. So assuming we know the critical values of our rejection region, we can calculate the power using the cdf and sf function of the binomial distribution.

# In[47]:

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


# In[48]:

for test, l, u in [('binom        ', 4, 15), ('binom_central', 3, 15), ('score        ', 4, 14)]:
    print(test, l, u, power_binom_reject(l, u, p_null, nobs))


# In[ ]:




# In[49]:

def power_binom_proptest(test_func, p_null, prop, nobs, alpha=0.05, args=(), kwds=None, item=None, use_idx=False):
    """calculate power for proportion test by explicit numeration of sample space
    
    
    argument `item` is currently to avoid having to figure out the return of test_func
    None if return is pvalue, integer for index of pvalue if tuple is returned
    
    """
    if kwds is None:
        kwds = {}
        
    sample_space = np.arange(nobs + 1)
    try:
        # TODO: how do we vectorize, if res were a instance with pvalue attribute, then it would be easier.
        res = test_func(sample_space, nobs, *args)
        #if len(res) > 1 and not res.shape == sample_space.shape:
            # assume p-value is the second term
        if item is not None:
            res = res[item]
    except Exception:
        # assume test_func is not vectorized
        if item is None:
            res = [test_func(x, nobs, p_null, *args, **kwds) for x in sample_space]
        else:
            res = [test_func(x, nobs, p_null, *args, **kwds)[item] for x in sample_space]
    
    pvalues = np.asarray(res)
    rej_indicator = (pvalues <= alpha)
    if use_idx:
        # This evaluates the pmf at all points, useful for non-interval rejection regions
        x_rej = sample_space[rej_indicator]
        power = stats.binom.pmf(x_rej, nobs, prop).sum()
        return power, x_rej
    else:
        # use critical values, assumes standard two tails, two-sided only for now
        c = np.nonzero(np.diff(rej_indicator))[0]
        if len(c) == 2:
            low = c[0]
            upp = c[1] + 1
        else:
            raise NotImplementedError('currently only two sided hypothesis tests')
            
        power = power_binom_reject(low, upp, prop, nobs)
        
        return power, (low, upp)
    


# We can use this function to check the size of the two binomial tests. Both results are what we already had before and agree with the results of R packages.

# In[50]:

print(power_binom_proptest(smprop.binom_test, p_null, p_null, nobs))
print(power_binom_proptest(smprop.binom_test, p_null, p_null, nobs, use_idx=True))
# 0.04709225  R library MESS: power.binom.test(n = 30, p0 = 0.3, pa = 0.3)


# In[ ]:




# In[51]:

print(power_binom_proptest(binom_test_centered, p_null, p_null, nobs))
print(power_binom_proptest(binom_test_centered, p_null, p_null, nobs, use_idx=True))
# 0.02625388 from exactci: powerBinom(n = 30, p0 = 0.3, p1 = 0.3, strict=TRUE)


# We obtain the power of the test at a proportion that is different from the proportion of the null hypothesis. Using the minlike binomial test the power if the true proportion is 0.5 is 0.57, the power for the central binomial test differs only in the 5th decimal from this.

# In[52]:

print(power_binom_proptest(smprop.binom_test, p_null, 0.5, nobs))
print(power_binom_proptest(smprop.binom_test, p_null, 0.5, nobs, use_idx=True))
# 0.572262  R library MESS: power.binom.test(n = 30, p0 = 0.3, pa = 0.5)


# In[53]:

print(power_binom_proptest(binom_test_centered, p_null, 0.5, nobs))
print(power_binom_proptest(binom_test_centered, p_null, 0.5, nobs, use_idx=True))
# 0.5722364 from exactci: powerBinom(n = 30, p0 = 0.3, p1 = 0.5, strict=TRUE)


# surprisingly this also works in vectorized for to calculate the power for a set of alternatives.

# In[54]:

p1 = np.linspace(0.1, 0.8, 15)
pbminlike = power_binom_proptest(smprop.binom_test, p_null, p1, nobs)
pbcentral = power_binom_proptest(binom_test_centered, p_null, p1, nobs)
pow_bt = np.column_stack((p1, pbminlike[0], pbcentral[0]))
pow_bt


# to check this let's use a list comprehension and explicitly loop over all alternative proportions

# In[55]:

[power_binom_proptest(smprop.binom_test, p_null, p1_, nobs) for p1_ in p1]


# And finally a plot.

# In[56]:

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(pow_bt[:, 0], pow_bt[:, 1], label='minlike')
plt.plot(pow_bt[:, 0], pow_bt[:, 2], label='central')
plt.legend(loc='lower right')
#plt.show()


# From the plot we can see that both binomial test have the same power for large true proportions, but the standard minlike binomial test is more powerful than the central binomial test for small true proportions. For example, if the true proportion is 0.15, then the probability of rejecting the null hypothesis are 0.52 versus 0.32. We can verify that the two R packages produce the same result

# In[57]:

# 0.5244758 power.binom.test(n = 30, p0 = 0.3, pa = 0.15)
# 0.321667  powerBinom(n = 30, p0 = 0.3, p1 = 0.15, strict=TRUE)
print(pow_bt[1,:])


# ### Power as a function of nobs

# In[58]:

nobs_arr = np.arange(30, 100)
#this doesn't work vectorized in nobs
pbcentral_nobs = [power_binom_proptest(binom_test_centered, p_null, 0.5, nobs_) for nobs_ in nobs_arr]
pbcentral_nobs


# In[59]:

pbminlike_nobs = [power_binom_proptest(smprop.binom_test, p_null, 0.5, nobs_) for nobs_ in nobs_arr]
pbminlike_nobs


# In[60]:

pbcentral_nobs_arr, rej_minlike = list(zip(*pbcentral_nobs))
pbcentral_nobs_arr
pbminlike_nobs_arr, rej_minlike = list(zip(*pbminlike_nobs))
np.column_stack((nobs_arr, pbminlike_nobs_arr, pbcentral_nobs_arr))


# In[61]:

plt.figure(figsize=(8, 6))
plt.plot(nobs_arr, pbminlike_nobs_arr, label='minlike')
plt.plot(nobs_arr, pbcentral_nobs_arr, label='central')
plt.legend(loc='lower right')


# In[62]:

xx = (np.arange(10)<4) | (np.arange(10) > 6)
print(xx)
np.nonzero(np.diff(xx))[0]


# In[63]:

p_null, nobs


# ## Power and tests based on normal distribution
# 
# 
# The following is still messy. The formulas look simple but are a bit confusing. There are also several different version for normal distribution based hypothesis tests and power calculations. The examples try to match up some examples from various references but that is not completely successful yet, either because of bugs in my code or because different versions are used.

# Lachine summarizes sample size calculations for proportions based on the normal distribution if we only consider the power in one tail. In this case we have an explicit formula for the required sample size. This is a good approximation to two sided tests if the probability to be in the small tail is negligible and useful for quick calculations. However, solving the sample size that correctly takes both tails into account can be done numerically without much computational effort.

# In[64]:

# from Lachine 1981 equ (3) and (4)

from scipy import stats
def sample_size_normal_greater(diff, std_null, std_alt, alpha=0.05, power=0.9):
    crit_alpha, crit_pow = stats.norm.isf(alpha), stats.norm.isf(1 - power)
    return ((crit_alpha * std_null + crit_pow * std_alt) / np.abs(diff))**2

def power_normal_greater(diff, std_null, std_alt, nobs, alpha=0.05):
    crit_alpha = stats.norm.isf(alpha)
    crit_pow = (np.sqrt(nobs) * np.abs(diff) - crit_alpha * std_null) / std_alt
    return stats.norm.cdf(crit_pow)


# In[65]:

pa = 0.5
power_normal_greater(pa - p_null, np.sqrt(p_null * (1 - p_null)), np.sqrt(pa * (1 - pa)), 30, alpha=0.05)


# In[66]:

std_null, std_alt = np.sqrt(p_null * (1 - p_null)), np.sqrt(pa * (1 - pa))
sample_size_normal_greater(pa - p_null, std_null, std_alt, alpha=0.05, power=0.7528)


# In[67]:

p0 = 0.6
pa = 0.5
power_normal_greater(pa - p0, np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa)), 25, alpha=0.05)


# In[68]:

p0 = 0.5
pa = 0.4
power_normal_greater(pa - p0, np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa)), 50, alpha=0.05)


# In[69]:

p0 = 0.3
pa = 0.5
diff = pa - p0
power_normal_greater(diff, np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa)), 50, alpha=0.05)


# In[70]:

p0 = 0.5
pa = 0.5
diff = pa - p0
diff = 0.2
power_normal_greater(diff, np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa)), 50, alpha=0.025)
# 0.80743 PASS manual example Chow, Shao, and Wang (2008)  2-sided S(Phat)


# In[71]:

p0 = 0.5
pa = 0.6
diff = pa - p0
power_normal_greater(diff, np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa)), 153, alpha=0.05)
# 0.80125 PASS doc example from Ryan (2013) for one-sided alternative


# In[72]:

# copied and adjusted from statsmodels.stats.power
def normal_power(effect_size, nobs, alpha, alternative='two-sided', std_null=1, std_alt=1):
    '''Calculate power of a normal distributed test statistic

    '''
    d = effect_size

    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.  #no inplace changes, doesn't work
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " +
                         "or 'smaller'")

    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit = stats.norm.isf(alpha_)
        pow_ = stats.norm.sf((crit* std_null - d*np.sqrt(nobs))/std_alt)
        crit_pow = (np.sqrt(nobs) * np.abs(diff) - crit * std_null) / std_alt
    if alternative in ['two-sided', '2s', 'smaller']:
        crit = stats.norm.ppf(alpha_)
        pow_ += stats.norm.cdf((crit* std_null - d*np.sqrt(nobs))/std_alt)
    return pow_ #, (crit* std_null - d*np.sqrt(nobs))/std_alt, (crit* std_null - d*np.sqrt(nobs))/std_alt, crit_pow


# In[73]:

p0 = 0.5
pa = 0.5
alpha = 0.05
nobs_ = 50
effect_size = diff = 0.2
std_null, std_alt = np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa))
po = normal_power(effect_size, nobs_, alpha, alternative='two-sided', std_null=std_null, std_alt=std_null)
print(po, 1-po)
# close to above 0.80742957881382105, closer to pass 0.80743


# In[74]:

p0 = 0.5
pa = 0.6
diff = pa - p0
effect_size = diff
nobs_ = 153
alpha = 0.05
std_null, std_alt = np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa))
po = normal_power(effect_size, nobs_, alpha, alternative='larger', std_null=std_null, std_alt=std_alt)
# 0.80125 PASS doc example from Ryan (2013) for one-sided alternative
print(power_normal_greater(diff, np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa)), 153, alpha=0.05))
po


# check size (power at null)

# In[75]:

p0 = 0.6
pa = 0.6
diff = pa - p0
effect_size = diff
nobs_ = 153
alpha = 0.05
std_null, std_alt = np.sqrt(p0 * (1 - p0)), np.sqrt(pa * (1 - pa))
po = normal_power(effect_size, nobs_, alpha, alternative='larger', std_null=std_null, std_alt=std_alt)
po


# In[ ]:




# Next we try exact power for the already available proportion_ztest

# In[76]:

p0 = 0.5
pa = 0.6
diff = pa - p0
smprop.proportions_ztest(nobs_ * (pa), nobs_, value=p0, alternative='two-sided', prop_var=p0)


# In[77]:

#power_binom_proptest(smprop.proportions_ztest, p0, pa, nobs_, use_idx=1)  #this raises exception

pzt = lambda x, nobs, p_null: smprop.proportions_ztest(x, nobs, value=p_null, prop_var=p_null)
power_binom_proptest(pzt, p0, pa, nobs_, item=1, use_idx=1)   #use_idx=False raises exception


# In[78]:

p0, pa, nobs_


# In[79]:

pv = [smprop.proportions_ztest(x, nobs_, value=p0, alternative='two-sided', prop_var=p0)[1] for x in np.arange(60, 99)]
pv = np.asarray(pv)
np.column_stack((np.arange(60, 99), pv, pv <=0.05))


# The power using the exact distribution is lower than using the asymptotic normal distribution.
# The rejection region looks correct, so how do we verify that we calculated the power correctly?
# 
# 
# PASS reports the following values
# 
# ```
#                           Exact  Z-Test  Z-Test  Z-Test  Z-Test
#                   Target   Test   S(P0)  S(P0)C    S(P)   S(P)C
# n    P0     P1    Alpha   Power   Power   Power   Power   Power
# 10 0.5000 0.6000 0.0500 0.04804 0.04804 0.04804 0.17958 0.17958
# 50 0.5000 0.6000 0.0500 0.23706 0.33613 0.23706 0.33613 0.23706
# ```

# In[80]:

p0, pa, nobs_ = 0.5, 0.6, 50
power_binom_proptest(pzt, p0, pa, nobs_, item=1, use_idx=1)


# 0.33613 is the same as reported by PASS for the exact power of the score test, `S(P0)`. Unfortunately for testing purposes, in this example Wald and score test report identical numbers for n=50.

# In[ ]:




# In[81]:

pzt_wald = lambda x, nobs, p_null: smprop.proportions_ztest(x, nobs, value=p_null, prop_var=None)
power_binom_proptest(pzt_wald, p0, pa, nobs_, item=1, use_idx=1)


# 10 0.5000 0.6000 0.0500 0.04804 0.04804 0.04804 0.17958 0.17958

# In[82]:

nobs_ = 10
pzt_wald = lambda x, nobs, p_null: smprop.proportions_ztest(x, nobs, value=p_null, prop_var=None)
power_binom_proptest(pzt_wald, p0, pa, nobs_, item=1, use_idx=1)


# This is the same as the Wald test, while the score test has much lower  power in this example. It is only around 0.048 which is the same in PASS and our calculations at the provided print precision.

# In[83]:

power_binom_proptest(pzt, p0, pa, nobs_, item=1, use_idx=1)


# Know we know how to use it, and I added keywords to the `power_binom_proptest` above, we can drop the use of lambda functions.

# In[84]:

power_binom_proptest(smprop.proportions_ztest, p0, pa, nobs_, item=1, use_idx=1)


# In[85]:

power_binom_proptest(smprop.proportions_ztest, p0, pa, nobs_, kwds={'prop_var': p0}, item=1, use_idx=1)


# In[86]:

print(power_binom_proptest(smprop.proportions_ztest, p0, pa, nobs_, item=1, use_idx=0))
print(power_binom_proptest(smprop.proportions_ztest, p0, pa, nobs_, kwds={'prop_var': p0}, item=1, use_idx=0))


# In[ ]:




# ## Sensitivity to misspecification
# 
# This is just a quick experiment.
# 
# 
# We go back to the exact binomial test in the standard minlike version with power evaluated using the exact distribution. The underlying assumption is that we have a set of independent Bernoulli experiments with identical probability of an event.
# 
# As a simple deviation we consider that we have 3 groups of observations with different true proportions. For the initial analysis we calculate the rejection rate, size and power, using Monte Carlo.
# 
# It looks like in this example with three fixed groups we have underdispersion, and the rejection ratio is lower than with a sinlge group. That means that in this case the binomial test is even more conservative than in the case where the binomial distribution is correctly specified. This is a surprising because unobserved heterogeneity and mixture distribution should lead to over dispersion, but we keep the composition of the population and of the sample fixed in this experiment and consequently do not get extra variation from a changing sample composition.
# 
# I had used equal group sizes in my intial choice of numbers for the Monte Carlo setup. That case did not show any overdispersion in the sampled proportions. This needs further investigation.

# In[87]:

smprop.binom_test(31, 60)


# In[88]:

power_binom_proptest(smprop.binom_test, 0.5, 0.5, 60)


# In[89]:

# our binomial sampling process
rvs = np.random.binomial(60, 0.5, size=10000)
m = rvs.mean()
m, rvs.var(), m / 60 * (1 - m / 60) * 60


# In[90]:

def binom_mix_rvs(size=1):
    #group fraction
    # np.random.multinomial(60, [1./3] * 3, size=size)
    # assume fixed population group size, instead of multinomial
    rvs1 = np.random.binomial([26, 20, 14], [0.33847, 0.5, 0.8], size=(size, 3))
    return rvs1.sum(1)


# In[91]:

# true binomial distribution

n_rep = 10000
res0 = np.empty(n_rep, float)
res0.fill(np.nan)
for i in range(n_rep):
    xc = np.random.binomial(60, 0.5)
    res0[i] = smprop.binom_test(xc, 60)

print((res0 < 0.05).mean())


# In[92]:

# mixed binomial distribution

n_rep = 10000
res0 = np.empty(n_rep, float)
res0.fill(np.nan)
for i in range(n_rep):
    xc = binom_mix_rvs()
    res0[i] = smprop.binom_test(xc, 60)

print((res0 < 0.05).mean())


# These two Monte Carlo experiments show that the rejection rate under the null hypothesis drops from 0.0276 to 0.0148. As expected, the rejection rate in the Monte Carlo corresponds closely to the exact power calculations which is 0.0273.
# 
# Below are some checks to see whether the random sampling works as expected.

# In[93]:

np.random.multinomial(60, [1./3] * 3, size=10)


# In[94]:

np.random.binomial([25, 10, 25], [0.4, 0.5, 0.6], size=(10, 3))


# In[95]:

np.random.binomial(20, [0.4, 0.5, 0.6], size=(1000, 3)).mean(0)


# In[96]:

rvs1 = binom_mix_rvs(size=100000)
m = rvs1.mean()
m, rvs1.var(), m / 60 * (1 - m / 60) * 60, (rvs1 / 60 * (1 - rvs1 / 60) * 60).mean()


# In[97]:

(np.array([26, 20, 14]) * [0.33847, 0.5, 0.8]).sum()


# In[ ]:




# ## Summary
# 
# Now, we have almost all the necessary pieces working and verified on a few example. The next step is to clean this up, convert it to usage friendly function or classes and convert the examples to unit tests.
# 
# We have now two exact hypothesis tests, `minlike` and `central`, two tests based on asymptotic normality, `wald` and `score`, and we have three ways of calculating the power, using the exact distribution, using the asymptotic normal distribution, and the already existing power calculation based on effect size that does not distinguish that variance is different under the null and under the alternative.
# 
# We are still missing some examples, power calculations for confidence intervals and equivalence tests, where some functions are already available in statsmodels.stats.proportions. We still need a function that finds the sample size given the functions for the power. 
# Vectorization for different alternatives or number of observations depends on the implementation details and does not work across all cases. 

# In[ ]:




# ## trying more
# 
# The rest below is just some unsorted experiments to try a few more things.

# In[ ]:




# TODO: The following is not correct because when we change the sample size, then the rejection region also changes.

# In[98]:

[power_binom_reject(4, 15, p_null, nobs_) for nobs_ in range(30, 50)]


# We can also calculate this in vectorized form for the set of sample sizes and all three tests:

# In[99]:

power_binom_reject(np.array([4, 3, 4]), np.array([15, 15, 14]), p_null, np.arange(30, 50)[:, None])


# In[ ]:




# In[ ]:




# ## Trying out two sample proportion, incorrect if nobs is scalar instead of same length as count.

# In[100]:

smprop.proportions_ztest(np.array([6,7]), nobs, value=0, alternative='two-sided', prop_var=p_null)


# In[101]:

smprop.proportions_ztest(np.array([6,7]), nobs*np.ones(2), value=1/30, alternative='two-sided', prop_var=p_null)


# In[102]:

smprop.proportions_ztest(np.array([6,7]), nobs, value=1/30, alternative='two-sided', prop_var=p_null)


# In[103]:

smprop.proportions_ztest(np.array([6,7]), nobs, value=-1/30, alternative='two-sided', prop_var=p_null)


# In[104]:

smprop.proportions_ztest(np.array([6,7]), nobs*np.ones(2), value=-1/30, alternative='two-sided', prop_var=p_null)


# In[ ]:




# In[105]:

#?smprop.proportion_confint()


# In[106]:

smprop.proportion_confint(count, nobs)


# In[107]:

from statsmodels.stats.proportion import proportion_effectsize
es = proportion_effectsize(0.4, 0.5)
smpow.NormalIndPower().solve_power(es, nobs1=60, alpha=0.05, ratio=0)
# R pwr 0.3447014091272153


# In[108]:

smpow.NormalIndPower().solve_power(proportion_effectsize(0.4, 0.5), nobs1=None, alpha=0.05, ratio=0, power=0.9)


# In[ ]:




# In[ ]:




# In[ ]:




# In[109]:

low, upp, nobs, p_alt = 0.7, 0.9, 509/2, 0.82
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.025, dist='norm',
                     variance_prop=None, discrete=True, continuity=0,
                     critval_continuity=0)
    


# In[110]:

low, upp, nobs, p_alt = 0.7, 0.9, 419/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm',
                     variance_prop=None, discrete=False, continuity=0,
                     critval_continuity=0)


# In[111]:

low, upp, nobs, p_alt = 0.7, 0.9, 417/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm',
                     variance_prop=None, discrete=False, continuity=1,
                     critval_continuity=0)


# In[112]:

low, upp, nobs, p_alt = 0.7, 0.9, 420/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='binom',
                     variance_prop=None, discrete=False, continuity=0,
                     critval_continuity=0)


# In[113]:

low, upp, nobs, p_alt = 0.7, 0.9, 414/2, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.025, dist='norm',
                     variance_prop=None, discrete=False, continuity=1,
                     critval_continuity=0)


# In[ ]:




# In[ ]:




# In[114]:

low, upp, nobs = 0.4, 0.6, 100
smprop.binom_tost_reject_interval(low, upp, nobs, alpha=0.05)


# In[115]:

value, nobs = 0.4, 50
smprop.binom_test_reject_interval(value, nobs, alpha=0.05)


# In[116]:

smprop.proportion_confint(50, 100, method='beta')


# In[117]:

low, upp, nobs = 0.7, 0.9, 100
smprop.binom_tost_reject_interval(low, upp, nobs, alpha=0.05)


# In[118]:

low, upp, nobs, p_alt = 0.7, 0.9, 100, 0.8
smprop.power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='binom',
                     variance_prop=None, discrete=False, continuity=0,
                     critval_continuity=0)


# In[119]:

low, upp, nobs, p_alt = 0.7, 0.9, 100, 0.8
smprop.power_binom_tost(low, upp, nobs, p_alt, alpha=0.05)


# In[120]:

low, upp, nobs, p_alt = 0.7, 0.9, 125, 0.8
smprop.power_binom_tost(low, upp, nobs, p_alt, alpha=0.05)


# In[ ]:




# In[ ]:




# In[ ]:




# In[121]:

# from Lachine 1981 equ (3) and (4)

from scipy import stats
def sample_size_normal_greater(diff, std_null, std_alt, alpha=0.05, power=0.9):
    crit_alpha, crit_pow = stats.norm.isf(alpha), stats.norm.isf(1 - power)
    return ((crit_alpha * std_null + crit_pow * std_alt) / np.abs(diff))**2

def power_normal_greater(diff, std_null, std_alt, nobs, alpha=0.05):
    crit_alpha = stats.norm.isf(alpha)
    crit_pow = (np.sqrt(nobs) * np.abs(diff) - crit_alpha * std_null) / std_alt
    return stats.norm.cdf(crit_pow)


# In[122]:

# Note for two sample comparison we have to adjust the standard deviation for unequal sample sizes
n_frac1 = 0.5
n_frac2 = 1 - n_frac1

# if defined by ratio: n2 = ratio * n1
ratio = 1
n_frac1 = 1 / ( 1. + ratio)
n_frac2 = 1 - n_frac1


# If we use fraction of nobs, then sample_size return nobs is total number of observations
diff = 0.2
std_null = std_alt = 1 * np.sqrt(1 / 0.5 + 1 / 0.5)
nobs = sample_size_normal_greater(diff, std_null, std_alt, alpha=0.05, power=0.9)
nobs


# In[123]:

#nobs = 858
power_normal_greater(diff, std_null, std_alt, nobs, alpha=0.05)


# In[124]:

alpha=0.05; power=0.9
stats.norm.isf(alpha), stats.norm.isf(1 - power)


# In[125]:

crit_alpha = stats.norm.isf(alpha)
(np.sqrt(nobs) * np.abs(diff) - crit_alpha * std_null) / std_alt


# In[126]:

stats.norm.cdf(_)


# In[127]:

smprop.binom_test_reject_interval([0.4, 0.6], [100], alpha=0.05)


# In[ ]:



