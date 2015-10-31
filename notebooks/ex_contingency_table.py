
# coding: utf-8

## Contingency Tables and Log-linear Models

# Examples for PR `#2418`

# Status:  Initial draft just to try out some examples for contingency tables and log-linear models. It is used to evaluate agreement and differences to SAS and R. For example SAS CATMOD uses Wald tests, SAS GENMOD uses LR tests. Currently, we don't have any automatic support for LR tests in GLM (and other models outside the linear regression models).
# 
# Spotchecking some of the results show good agreement with R or SAS, but this notebook is not organized yet.

# In[1]:

import numpy as np
import pandas as pd
from statsmodels import datasets
import statsmodels.stats.api as smstats
import statsmodels.stats.contingency_tables as ctab

from numpy.testing import assert_allclose


### Example 1: 2 x k table

# In[2]:

df = datasets.get_rdataset("Arthritis", "vcd").data
tab = pd.crosstab(df['Treatment'], df['Improved'])
table = smstats.Table(tab)


# In[2]:




# In[3]:

table


# In[4]:

#print(table.summary())


# In[5]:

table.table


# In[6]:

vars(table.nominal_association)


# In[7]:

table.pearson_resids


# In[8]:

table.fittedvalues


# In[9]:

#table.resids


# In[10]:

table.fittedvalues


# In[11]:

table.local_oddsratios


# In[12]:

table.cumulative_log_oddsratios


# In[13]:

vars(table.ordinal_association())


# In[13]:




# In[13]:




# In[14]:

tab1 = [[[8, 9], [6, 7]], [[4, 9], [5, 5]], [[8, 8], [9, 11]]]
tab2 = np.asarray(tab1).T

ct1 = ctab.StratifiedTables(tab1)
ct2 = ctab.StratifiedTables(tab2.T)

assert_allclose(ct1.oddsratio_pooled, ct2.oddsratio_pooled)
assert_allclose(ct1.logodds_pooled, ct2.logodds_pooled)


# In[15]:

dir(ctab)


# In[16]:

tab1


# In[17]:

tab2.shape


# In[18]:

ct1.oddsratio_pooled, ct2.oddsratio_pooled


# In[18]:




# In[18]:




### Example 2x2 Loglinear Model (GLM)

# This replicates the SAS numbers from a 2 by 2 Vitamin C example at https://onlinecourses.science.psu.edu/stat504/node/120

# In[19]:

table_ski = [[31, 109], [17, 122]]


# In[20]:

count = np.array(table_ski).flatten()


# In[21]:

rows = np.repeat(['Placebo', 'Ascorbic Acid'], 2)
cols = np.tile(['cold', 'no cold'], 2)


# In[22]:

count


# In[23]:

from statsmodels.discrete.discrete_model import Poisson


# In[24]:

saturated = 'count ~ rows + cols + rows * cols - 1'
modp = Poisson.from_formula('count ~ rows + cols', data={'rows':rows, 'cols':cols, 'count':count})
# using only default optimizer newton doesn't work. Use Nelder-Mead first
resp = modp.fit(method='nm')
resp = modp.fit(start_params=resp.params, method='newton')
print(resp.summary())


# In[25]:

modp.exog


# In[26]:

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families


# In[27]:

mod = GLM.from_formula('count ~ rows + cols', data={'rows':rows, 'cols':cols, 'count':count}, family=families.Poisson())
res = mod.fit()
print(res.summary())


# In[28]:

resp.fittedvalues


# In[29]:

ct = smstats.Table(table_ski)


# In[30]:

ct.fittedvalues


# In[31]:

res.fittedvalues


# In[32]:

ct.pearson_resids


# In[33]:

resp.resid / resp.fittedvalues


# In[34]:

res.resid_pearson


# In[35]:

print(res.summary())


# In[36]:

res.wald_test_terms()


# In[37]:

res.null_deviance


# In[38]:

res.null


# In[39]:

from scipy import stats
res.deviance, stats.chi2.sf(res.deviance, 1)  # 


# In[40]:

# I needed to disable the PerfectSeparationError in the source for the saturated model, 
# see https://github.com/statsmodels/statsmodels/issues/2680

mod = GLM.from_formula('count ~ rows + cols + rows * cols', data={'rows':rows, 'cols':cols, 'count':count}, family=families.Poisson())
res = mod.fit()
print(res.summary())


# In[41]:

res.wald_test_terms()


# SAS CATMOD has the same value for the interaction term `rows:cols 4.69 0.0303 1` but different values for the other two terms `rows` and `cols`. CATMOD uses zero sum coding, but that shouldn't change whether the effect is zero, although it changes the parameterization.

# In[42]:

res.wald_test_terms(combine_terms = ['rows', 'cols'])


# In[43]:

res.model.exog_names


# In[44]:

[i for i in res.model.exog_names if 'cold' in i]


# In[44]:




### Example 3: Three Factors - Berkeley admission

# See for example https://en.wikipedia.org/wiki/Simpson%27s_paradox#Berkeley_gender_bias_case for the gender discrimination issue.
# 
# 

# In[45]:

columns = 'D S A count'.split()

ss = '''Dep Sex Admit count
DeptA  Male    Reject  313
DeptA  Male    Accept  512
DeptA  Female  Reject   19
DeptA  Female  Accept   89
DeptB  Male    Reject  207
DeptB  Male    Accept  353
DeptB  Female  Reject    8
DeptB  Female  Accept   17
DeptC  Male    Reject  205
DeptC  Male    Accept  120
DeptC  Female  Reject  391
DeptC  Female  Accept  202
DeptD  Male    Reject  278
DeptD  Male    Accept  139
DeptD  Female  Reject  244
DeptD  Female  Accept  131
DeptE  Male    Reject  138
DeptE  Male    Accept   53
DeptE  Female  Reject  299
DeptE  Female  Accept   94
DeptF  Male    Reject  351
DeptF  Male    Accept   22
DeptF  Female  Reject  317
DeptF  Female  Accept   24'''


# In[46]:

import io  # use compat
df_adm = pd.read_csv(io.StringIO(ss), delim_whitespace=True)


# In the first case we only use main effects, which assumes that the factors are independent. 

# In[47]:

mod = GLM.from_formula('count ~ Dep + Sex + Admit', data=df_adm, family=families.Poisson())
res = mod.fit()
print(res.summary())


# In[48]:

mod = GLM.from_formula('count ~ Dep + Sex + Admit + Dep*Sex', data=df_adm, family=families.Poisson())
res = mod.fit()
print(res.summary())


# In[49]:

res.wald_test_terms()


# In[50]:

mod = GLM.from_formula('count ~ Dep + Sex + Admit + Dep*Sex + Sex*Admit', data=df_adm, family=families.Poisson())
res = mod.fit()
print(res.summary())
res.wald_test_terms()


# In[51]:

mod = GLM.from_formula('count ~ Dep + Sex + Admit + Dep*Sex + Sex*Admit + Dep*Admit', data=df_adm, family=families.Poisson())
res = mod.fit()
print(res.summary())
res.wald_test_terms()


# In[52]:

print(res.wald_test_terms(combine_terms=['Sex']))


# In[53]:

deviance_pairs = res.deviance
df_resid_pairs = res.df_resid


#### Comparison with Binomial

# In[54]:

help(pd.pivot)


# In[56]:




# In[57]:

t = pd.pivot_table(df_adm, index=['Dep', 'Sex'], columns=df_adm[['Admit']], values='count')


# In[58]:

type(t)


# In[59]:

t.index


# In[60]:

t['total'] = t.sum(1)


# In[61]:

t.columns


# In[63]:

t2 = t.reset_index()


# In[64]:

t2


# In[65]:

endog = t2[['Accept', 'total']]
#exog = ?


# In[66]:

modb = GLM.from_formula('Accept + total ~ Dep + Sex', data=t2, family=families.Binomial())
resb = modb.fit()
print(resb.summary())
resb.wald_test_terms()


# In[67]:

modb = GLM.from_formula('Accept + total ~ Dep + Sex + Dep*Sex', data=t2, family=families.Binomial())
resb = modb.fit()
print(resb.summary())
resb.wald_test_terms(combine_terms=['Sex'])


# In[68]:

# resb.wald_test_terms(combine_terms=['Male']) # exception


# In[69]:

formula_2int = 'count ~ Dep + Sex + Admit + Dep*Sex + Sex*Admit + Dep*Admit'
mod = GLM.from_formula(formula_2int + ' - Dep*Admit', data=df_adm, family=families.Poisson())
res = mod.fit()
print(res.summary())
res.wald_test_terms()


# In[70]:

formula_2intc = 'count ~ Dep + Sex + Admit + Dep*Sex + Sex*Admit'
mod = GLM.from_formula(formula_2intc, data=df_adm, family=families.Poisson())
res = mod.fit()
print(res.summary())
res.wald_test_terms()


# In[71]:

interact2way = 'Dep*Sex Sex*Admit Dep*Admit'.split()
mod = GLM.from_formula('count ~ Dep + Sex + Admit', data=df_adm, family=families.Poisson())
res = mod.fit()
print(res.summary())
res.wald_test_terms()


# In[72]:

import itertools
list(itertools.combinations(range(3), 2))


# **Type 3 Analysis of Deviance, Likelihood Ratio tests**

# In[73]:

interact2way = 'Dep*Sex Sex*Admit Dep*Admit'.split()


# In[74]:

for i, j in itertools.combinations(range(3), 2):
    #print(interact2way[i], interact2way[j], end=' ')
    formula_i = ' + '.join(['count ~ Dep + Sex + Admit', interact2way[i], interact2way[j]])
    #print(formula_i, end=' :   ')
    print('drop ' + ' '.join(interact2way[k] for k in sorted(list(set(range(3)) - set((i,j))))), end=' :   ')
    res_ij = GLM.from_formula(formula_i, data=df_adm, family=families.Poisson()).fit()
    dev_ij = res_ij.deviance
    
    print(dev_ij, dev_ij - deviance_pairs, res_ij.df_resid - df_resid_pairs,  
          stats.chi2.sf(dev_ij - deviance_pairs, res_ij.df_resid - df_resid_pairs))


# In[75]:

deviance_pairs


# see https://onlinecourses.science.psu.edu/stat504/node/129 for some related deviance numbers and a full table (to be filled out).

# In[75]:



