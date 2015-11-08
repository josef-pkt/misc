
# coding: utf-8

## Log-linear Models

# This is an example from Agresti: *Categorical Data Analysis*, 2nd edition, 2003.

# In[1]:

import numpy as np
from scipy import stats
import pandas as pd
from statsmodels import datasets
import statsmodels.stats.api as smstats
import statsmodels.stats.contingency_tables as ctab

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links

from numpy.testing import assert_allclose


# In[2]:

from cda_data import df_injury, df_injury_bin
df_injury


# In[3]:

df_injury_bin['total']  = df_injury_bin['no_inj'] + df_injury_bin['inj']

mod_b = GLM.from_formula('inj + no_inj ~ belt + location + gender', data=df_injury_bin, family=families.Binomial())
res_b = mod_b.fit()
print(res_b.summary())
#resb.wald_test_terms()
print(res_b.fittedvalues * df_injury_bin['total'])   # fitted count


# In[4]:


mod_main = GLM.from_formula('count ~ injury + belt + location + gender', data=df_injury, family=families.Poisson())
res_main = mod_main.fit()
print(res_main.summary())


# Copy the original data frame so we can add results.

# In[5]:

df_res_injury = df_injury.copy()


# In[6]:

df_res_injury['fitted_main'] = res_main.fittedvalues
df_res_injury['resid_pearson_main'] = res_main.resid_pearson
df_res_injury


# Note that the Pearson chisquare statistic is very large, the residuals are also very large. This simple model of complete independence fits very badly. Let's try all pairwise interactions with injury. 

# In[7]:

mod_I2 = GLM.from_formula('count ~ injury + belt + location + gender + injury * (belt + location + gender)', 
                          data=df_injury, family=families.Poisson())
res_I2 = mod_I2.fit()
print(res_I2.summary())
print(res_I2.wald_test_terms())


# In[8]:

res_I2.resid_pearson


# In[9]:

mod_I23 = GLM.from_formula('count ~ injury + belt + location + gender + injury * (belt + location + gender) + belt * location * gender', 
                          data=df_injury, family=families.Poisson())
res_I23 = mod_I23.fit()
print(res_I23.summary())
print(res_I23.wald_test_terms())


# In[10]:

df_res_injury['fitted_23'] = res_I23.fittedvalues
df_res_injury['resid_pearson_23'] = res_I23.resid_pearson
df_res_injury


# This model fits well, neither pearson chisquare not likelihood ration test (deviance) rejects the null of this reduced model compared to the saturated model.

# In[11]:

res_I23.pearson_chi2, stats.chi2.sf(res_I23.pearson_chi2, res_I23.df_resid), res_I23.df_resid


# In[12]:

res_I23.deviance, stats.chi2.sf(res_I23.deviance, res_I23.df_resid), res_I23.df_resid


# Following the convention in loglinear analysis with many interaction effects, we can use shortened names for more compact representation. 

# In[13]:

data = df_injury.copy()
data.columns = 'B L G I count'.split()


# In[14]:

data.head()


# As an example for selecting a model we consider the case with all two way interactions

# In[15]:

formula_main = 'count ~ I + B + L + G'
extras = 'I:B I:L I:G B:L B:G L:G'.split()
formula_i = ' + '.join([formula_main] + extras)
mod_i = GLM.from_formula(formula_i, data=data, family=families.Poisson())
res_i = mod_i.fit()

#print(res_i.summary())
print(formula_i)
print()
print(res_i.wald_test_terms())
print()
print('Pchi2 :', res_i.pearson_chi2, stats.chi2.sf(res_i.pearson_chi2, res_i.df_resid), res_i.df_resid)
print('LR :   ', res_i.deviance, stats.chi2.sf(res_i.deviance, res_i.df_resid), res_i.df_resid)


# In[16]:

#print(res_i.summary())


# Next we consider the model with all three-way interactions. All terms are significant, the Pearson chisquare and the LR test do not reject the restricted model compared to the saturated model.

# In[17]:

formula_main = 'count ~ I + B + L + G'
extras = 'G:I:L G:I:B G:L:B I:L:B'.split()
formula_i = ' + '.join([formula_main] + extras)
mod_i = GLM.from_formula(formula_i, data=data, family=families.Poisson())
res_i = mod_i.fit()

#print(res_i.summary())
print(formula_i)
print()
print(res_i.wald_test_terms())
print()
print('Pchi2 :', res_i.pearson_chi2, stats.chi2.sf(res_i.pearson_chi2, res_i.df_resid), res_i.df_resid)
print('LR :   ', res_i.deviance, stats.chi2.sf(res_i.deviance, res_i.df_resid), res_i.df_resid)


# Agresti reports the deviance and associated LR pvalues in Table 8.1 for the 4 models above and several additional models. The only two models that are acceptable based on the Pearson chisquare and likelihood ratio tests are the model with all three-way interactions and the model with all two-way interactions with injury and three way interaction for non-injury variables. The former has only one residual degree of freedom, the latter has 4

# The Binomial model has a corresponding loglinear model that produces identical effects of the explanatory variables on the *endogenous* variable 

# In[18]:

pI23 = res_I23.params[[i for i in res_I23.params.index if 'injury[T.Yes]:' in i]]


# In[19]:

pbinom = res_b.params[1:]


# In[20]:

print(pI23.values)
print(pbinom.values)


# In[21]:

pI23


# In[22]:

pbinom


# The fit statistics, Pearson chisquare and deviance are also identical in the two models, up to numerical precision.

# In[23]:

res_I23.deviance, res_b.deviance


# In[24]:

res_I23.pearson_chi2, res_b.pearson_chi2


# In[24]:



