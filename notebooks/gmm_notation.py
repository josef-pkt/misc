
# coding: utf-8

## GMM - Notation

# Author: Josef Perktold

# TODO: check shapes for arrays in math: column or row vectors, and transpose

### The general setup

# The following follows the notation for GMM in McFadden's lecture notes. It assumes iid observations, but large part generalize with appropriate redefinition of the covariance of the moment conditions. (One problem with McFaddens notation is that capital letter of a lower case name of a function refers to the derivative, while for us it would be more useful to refer the the array or vector of functions over observations.)
# 
# Letters with subscript `n` denote empirical means, i.e. expectation with respect to the empirical distribution. Letters without subscripts refer in most cases to expectation with respect to the true, unknown distribution. Since we are not deriving any asymptotic results, we will drop the `n` subscript also for the estimate or empirical value.
# 
# Note or Todo: For simplicity we ignore in most of the following the normalization by the number of observations $1 / n$. This is important for the asymptotic analysis, but in most cases the normalization drops out of the quadratic forms.

# objective function:  $$Q(\theta) = 0.5 g_n(\theta)' W_n (\tau_n) g_n(\theta)$$
# 
# moment condition   $$g_n(\theta) = \sum_i{g(z_t, \theta)}$$
# 
# jacobian of moment condition  $$G_n(\theta) = - \frac{1}{n} \sum_i{\Delta_\theta g(z_t, \theta)}$$
# 
# covariance of moment conditions (i.i.d.)   $$S_n(\theta) = \Omega_n(\theta) = \frac{1}{n} \sum_i{g(z_t, \theta) g(z_t, \theta)'}$$
# 
# For this covariance we will use $S$ as an alias in the following.
# 

# From now on we drop subscript $n$ and do not explicitly include the dependence on $theta$ unless we need it. In the following we will have several quadratic forms that are repeatedly used.
# 
# \begin{align}
# B &= G' S G \\
# C &= G' W G\\
# H &= G' W S W G
# \end{align}
# 
# 
# The covariance estimate of the parameters assuming efficient GMM, $W = S$ is $$(G' S G)^{-1}.$$
# For arbitrary weight matrix, the covariance is $$(G' W G)^{-1} G' W S W G (G' W G)^{-1}$$ or $$C^{-1} H C^{-1}.$$
# 
# 

### Iteration

# The following is a detour to describe briefly the implementation in statsmodels related to the iterative updating of the weight matrix and estimation of the parameter and their covariance.
# 
# (TODO: figure out formatting for this section)
# 
# This is based on what I remember and needs to be checked with the code, especially for maxiter=0.
# 
# Define $W^{(0)}$ as the initial weight matrix, which is the identity matrix by default, and denote by parenthesis in superscript the value after an iteration.
# 
# The covariance of the parameters is
# 
# maxiter = 0, efficient=True :   $$(G(\theta^{(1)})' \hspace{3pt} W^{(0)} \hspace{3pt} G(\theta^{(1)}))^{-1}$$
# 
# **check:** maxiter=0 needs another review, maybe not correct for the general case, we still need scale in formula for OLS case
# 
# maxiter = 1, efficient=True :   $$(G(\theta^{(1)})' \hspace{3pt} S(\theta^{(1)}) \hspace{3pt} G(\theta^{(1)}))^{-1}$$
# 
# Note: this assumes that $E W^{(0)} = E S(\theta^{(1)}) = \Omega$ for the asymptotic analysis.
# 
# maxiter = 0, efficient=False :   $$(G' W G)^{-1} \hspace{3pt} G' W S W G \hspace{3pt} (G' W G)^{-1}$$ 
# 
# where $G$ and $S$ are evaluated at $\theta^{(1)}$, i.e. $W = W^{(0)}$, $S = S(\theta^{(1)})$ and $G = G(\theta^{(1)})$
# 
# 
# For iterative GMM after iteration $i$ we have for efficient GMM :   
# $$(G(\theta^{(i)})' \hspace{3pt} S(\theta^{(i)}) \hspace{3pt} G(\theta^{(i)}))^{-1}$$
# 
# In each case, $\theta^{(i)}$ at $i=1, 2, 3, ...$ solves
# 
# $$\theta^{(i)} = \text{argmin}_{\theta} \hspace{5pt} 0.5 \hspace{3pt} g_n(\theta)′ \hspace{3pt} W_n(\theta^{(i-1)}) \hspace{3pt} g_n(\theta)$$
# 
# GMM with continuous updating treats the weight matrix also as a function of the parameters, i.e.
# 
# $$\theta^{cu} = \text{argmin}_{\theta} \hspace{5pt} 0.5 \hspace{3pt} g_n(\theta)′ \hspace{3pt} W_n(\theta) \hspace{3pt} g_n(\theta)$$

### Special Case: Exactly Identified Models

# If we have as many moment conditions as unknown parameters, then the parameters are exactly identified, under rank conditions additional to the usual identification assumptions. In this case we can solve the moments exactly for the parameters, all moment conditions will be zero at the optimum, and the solution does not depend on the weight matrix.
# 
# It also implies that $G$ is a full rank square matrix and we can split the inverse of quadratic forms into quadratic forms of the inverses, for example 
# $$(G' S G)^{-1} = (G')^{-1} S^{-1} G^{-1}$$ and
# $$(G' W G)^{-1} = (G')^{-1} W^{-1} G^{-1}$$.
# 
# If we use $W = G^{-1}$ when $G$ is symmetric, then the covariance
# 
# $$(G' W G)^{-1} \hspace{3pt} G' W S W G \hspace{3pt} (G' W G)^{-1}$$ 
# 
# becomes
# $$(G')^{-1} G G^{-1} \hspace{3pt} G' G^{-1} S G^{-1} G \hspace{3pt} (G')^{-1} G G^{-1}$$ 
# 
# and
# $$(G')^{-1} S G^{-1}$$
# 
# If we use identity weight matrix, then this becomes
# $$(G')^{-1} G^{-1} \hspace{3pt} G' S G \hspace{3pt} (G')^{-1} G^{-1}$$ 
# 
# which also reduces to
# $$(G')^{-1} S G^{-1}$$
# 
# 
# This last expression is the standard form of a sandwich covariance where in the linear model $G$ is given by $X'X$ for exogenous variables  $X$.

# In[ ]:




### Special Case: Nonlinear IV-GMM

# In the general setup of the GMM estimation, we have not imposed any specific structure on the moment conditions. In a very large class of models we can separate the moment conditions into two multiplicative parts, the instruments and a *essential zero function*:
# $$ g(y, x, z)  = z f(y, x, \theta) $$
# 
# In the simple case the essential zero function $f(y, x, \theta)$ is a scalar function that multiplies each element of the instruments $z$.
# 
# This case makes it easier to obtain additional results and it provides some simplification in the definition and implementation of the model and the estimator. The linear model where f is the residual of a linear function is just a special case of this where the main advantages for us are in using linear algebra to get an explicit solutions of the optimization problem, instead of using a nonlinear optimization algorithm.
# 
# (Footnote: Instruments and IV in the current general context does not refer specifically to the endogeneity problem, it is just a name for the multiplicative variables in the moment conditions. Instruments can be for example the explanatory variables if there are no endogenous variables, they can be non-linear functions of the explanatory variables can also include other linear transformation of the moment conditions conditions.)
# 
# The following list shows essential zero function $f(y, x, \theta)$ for some common models.
# 
# In both linear model and nonlinear least squares models it is given by the residuals
# 
# $$f(y, x, \theta) = y - X \theta$$
# 
# and
# $$f(y, x, \theta) = y - m(X, \theta)$$
# 
# Models with specified heteroscedasticity and generalize linear model have weighted residuals
# 
# $$f(y, x, \theta) = \frac{y - m(x \theta)}{V(x, \theta, ...)}$$
# 
# where $V(x, \theta, ...)$ can depend on the explanatory variables, on the predicted mean and on variance specific exogenous variables.
# 

# In[ ]:




# In[ ]:




### Hypothesis testing

# We have the three main hypothesis tests available to test hypotheses on the values of or restrictions on parameter, the Wald test, the score or Lagrange multiplier test and the j-test that compares the values of the objective function which is the GMM analog to the likelihood ratio test.
# 
# Wald tests are easy to implement and are generally available in all statistical packages since they only require the asymptotic covariance matrix of the parameters. Wald tests are based on unrestricted parameter which additionally allows to test different hypothesis without estimating the model under different restrictions. An additional advantage of Wald tests is that it is relatively easy to obtain robust covariance matrices, i.e. sandwich covariance of the parameters that are robust to unspecified correlation and heteroscedasticity.
# The main disadvantage of Wald tests is that they are often liberal in small samples and reject too often, in some case they can be very liberal.
# 
# The likelihood ration test requires that the entire likelihood function is correctly specified, with a few exceptions like over or underdispersion that can be captured by a constant scale factor. Similarly, the J-test only has asymptotically a chisquare distribution if we use efficient GMM, i.e. the weight matrix is (asymptotically) equal to the covariance of the moments conditions.
# A technical complication is that it is generally recommended that the weight matrix is kept constant when calculating the value of the objective function under the restricted and under the unrestricted model. 
# Note: Davidson and MacKinnon state that this is a requirement for having a chisquare distribution, but I don't see that yet. It shouldn't matter the first order asymptotics under the null. which approximation for the covariance of moment restrictions we use as weight matrix. However, in small samples, it is possible that the difference in the objective function has the wrong sign if we use different weight matrices.
# 
# Finally, the score or Lagrange multiplier test require that we estimate the restricted model under the null hypothesis. Estimating the restricted models can be easier when we want to compute diagnostic tests to check whether the restricted model is correct, but it requires that we estimated each restricted model separately if we test several restrictions, as in the use case for the Wald test that provides p-values and confidence intervals for each parameter. 
# 
# In many cases, score test are conservative and often they are closer to having correct size than Wald tests in small samples. All three tests are asymptotically equivalent and have the same limiting distribution, with the restriction that J-test or LR test does not apply in all cases for which Wald and score tests are available. Additionally, in many cases there are different ways of calculating the score or wald statistic that are all asymptotically equivalent. However, the tests and versions of the tests have different accuracy in small samples and differ in general in higher order asymptotics. 
# 
# In some cases we have a good indication about small sample properties which often comes from Monte Carlo studies. In those cases we can focus on implementing better behaved statistics if that is computationally feasible. 
# 
# In the following we mainly look at score or Lagrange Multiplier test. They are currently still mostly missing in statsmodels as a general approach to hypothesis testing.
# 
# 

# The general nonlinear hypothesis that we want to test is $a(\theta) = 0$ with gradient function $A(\theta) = \Delta_{\theta} a(\theta)$
# 
# Common special cases are linear and affine restrictions:
# $$A \theta = 0\\
# A \theta - b = 0
# $$
# 
# and zero restrictions
# $$ \theta_i = 0 $$
# 
# where $\theta_i$ is an element or a subset of elements of $\theta$.

#### Score or Lagrange Multiplier tests

# We can compute the constrained model either through reparameterization or through explicitly solving a constrained optimization problem. The former is especially convenient in diagnostic tests where the constrained model is easier to estimate than a fully specified unrestricted problem. As example consider testing for heteroscedasticity where the full model requires the specification and estimation of a model that incorporates the variance function. Testing Poisson against an model that allows over or underdispersion requires that we specify and estimate a model that allows for both. In both cases it is easy to estimate the restricted model. In most cases only the reparameterized version is implemented in a statistical package.
# 
# The equations for the Lagrange multiplier tests can be confusing because we need to distinguish between moment conditions that are used in a constrained reparameterized model and the model that would be obtained by imposing the restrictions on a fully specified unrestricted model.
# 
# Conditional moment tests and score tests for non GMM models raise a similar issue, see ....

# In[1]:

'''
todo

When we estimate based on a subset of moment conditions:
McFadden has a section on overidentifying restriction and reformulation as LM test. I used the idea in the GMM-GLM notebook. 
That trick might be useful as general approach to implementing extra moment conditions. (see notes in issue)
Also, specification tests as special case, e.g. heteroscedasticity.

On version of LM test in the GMM-GLM notebook only works in exactly identified models, g V g, general version coming up. 
It looks like I'm going to reimplement conditional moment tests directly as a GMM LM test. 
Robust (HC, HAC, cluster, ...) LM test will then be "for free".

I'm getting closer to understanding the variations of score or LM tests.

Essentially, I need to implement two version, one with arbitrary restrictions on parameters 
(a general fit_constrained when the full model is available) 
and a version that tests additional moment conditions added to an estimated restricted model.

Variation: Testing individual moment condition when we add several. Robust specification tests. 
papers ??? (I read them during conditional moment test development)


'''
0


# The following summarizes the LM test statistics in McFadden. In these formulas, capital letters without subscript refer to asymptotic quantities that can be estimated using the constrained parameters. However, under the null hypothesis different estimates all converge to the same values.
# 
# The Lagrange multipliers of the constrained GMM optimization are denoted by $\gamma_{an}$. All other terms are defined above. Superscript minus is used for the generalized inverse, the Moore-Penrose inverse.
# 
# The first set of LM test statistics are for efficient GMM, i.e. with $W = S^{-1}$
# 
# $LM_{1n}$ $$n \hspace{3pt} \gamma_{an} \hspace{3pt} A B^{-1} A' \hspace{3pt} \gamma_{an}$$
# 
# $LM_{3n}$ $$n \hspace{3pt} \Delta_{\theta} Q_n(T_{an})' \hspace{3pt} B^{-1} \hspace{3pt} \Delta_{\theta} Q_n(T_{an})$$
# 
# $LM_{2n}2$ $$n \hspace{3pt} \Delta_{\theta} Q_n(T_{an})' \hspace{3pt} [ A'  \hspace{3pt}(A B^{-1} A')^{-1} \hspace{3pt} A']^- \hspace{3pt} \Delta_{\theta} Q_n(T_{an})$$
# 
# $LM_{2n}1$ $$n \hspace{3pt} \Delta_{\theta} Q_n(T_{an})' \hspace{3pt} B^{-1} A'  \hspace{3pt}(A B^{-1} A')^{-1} \hspace{3pt} A B^{-1} \hspace{3pt} \Delta_{\theta} Q_n(T_{an})$$
# 
# The score or derivative of the GMM objective function is
# $$\Delta_{\theta} Q_n(T_{an}) = G W g(z, T_{an})$$
# 
# substituting this and the definiton of $B$ into $LM_{3n}$ and using $W = S^{-1}$, it becomes
# $$n \hspace{3pt}  g(z, T_{an})' S^{-1} G'\hspace{3pt} (G' S^{-1} G)^{-1} \hspace{3pt} G S^{-1} g(z, T_{an})$$
# 
# In the exactly identified case G is square and invertible and we can factor the inverse matrices to obtain
# $$n \hspace{3pt}  g(z, T_{an})' S^{-1} G' G'^{-1} S G^{-1}  G S^{-1} g(z, T_{an})$$
# 
# which simplifies to
# $$n \hspace{3pt}  g(z, T_{an})' S^{-1} g(z, T_{an})$$
# 
# which is the same as the score statistic for maximum likelihood models where g is the score function, that is the first derivative of the loglikelihood function, and S is ???.
# 
# We will continue later with this form of the score test for the special case of specification testing when the constrained model has been estimated with a subset of moment restrictions and we want to test the specification by adding additional moment conditions. If the initial model is exactly identified as in MLE models, then the initial set of moment conditions, the initial subvector of $g$, will be zero. 

# If we do not use efficient GMM and the weight matrix is not the inverse of the covariance matrix of the moment conditions, then terms in the quadratic forms do not cancel and we need the long version. $LM_{3n}$ is not available in this case because it does not have a asymptotic chisquare distribution.
# 
# $LM_{1n}$ $$n \hspace{3pt} \gamma_{an} \hspace{3pt} A C^{-1} A' \hspace{3pt} [A C^{-1} H C^{-1} A']^{-1} \hspace{3pt} A C^{-1} A' \hspace{3pt} \gamma_{an}$$
# 
# $LM_{2n}1$ $$n \hspace{3pt} \Delta_{\theta} Q_n(T_{an})' \hspace{3pt} [A' (A C^{-1} A')^{-1} \hspace{3pt} A C^{-1} H C^{-1} A' \hspace{3pt} (A C^{-1} A')^{-1} A]^{\mathbf{-}} \hspace{3pt} \Delta_{\theta} Q_n(T_{an})$$
# 
# $LM_{2n}2$ $$n \hspace{3pt} \Delta_{\theta} Q_n(T_{an})' \hspace{3pt} A' \hspace{3pt} [A C^{-1} H C^{-1} A']^{-1} \hspace{3pt} A  \hspace{3pt} \Delta_{\theta} Q_n(T_{an})$$
# 
# 
# **Implementation note:** The last version is relatively simple, the center part $C^{-1} H C^{-1}$ is the covariance of the parameter estimate in the unconstrained model. However, this is not directly available in the current implementation for the restricted model, where the underlying matrices are evaluated at the constrained parameter estimates.
# It would be easy to implement if `maxiter` had an option to not update or estimate the parameter. Otherwise we have to evaluate all underlying matrices directly.
# 
# All version of the LM test statistic are asymptotically equivalent, including variations that use different estimates for the matrices in the quadratic forms that are consistent under the Null (and local alternatives). However, small sample or higher order properties will differ across variations, but I do not know in which way.
# 
# 
# 
# 
# 

# 

#### Specification Testing in exactly identified models

# This is the part that got me initially interested in using GMM for implementing score tests. 
# 
# A large number of specification tests are available for the linear model, where we estimate an OLS model and then test for various deviations from the intitial specification, for example we test for omitted variables, heteroscedasticity, correlation and nonlinearity.
# Under normality, and in general with all distributions in the linear exponential family, the parameters for the covariance of the error are asymptotically uncorrelated with the mean paramters. This block structure provides a very simple form of to test for heteroscedasticity and correlation because we do not need to take the interaction of mean and covariance parameters into account in the Lagrange multiplier tests.
# 
# The following illustrates two examples from the section on overidentifying restrictions in McFadden.

# **Example 1 **
# 
# In the homoscedastic linear model $y = x \beta + u$ with $E(y|x) = 0$ and $E(u|x) = \sigma^2$ the moment conditions are
# 
# \begin{align}
# &x (y - x \beta) \\
# &(y - x \beta)^2 - \sigma^2
# \end{align}
# 
# If the errors $u$ are conditonally normally distributed $u \sim N(0, \sigma^2)$, then we have additional restrictions on higher order moments. Normality tests like Jarque Bera test restrictions on the values of skew and kurtosis or third and fourth moment. Those moment conditions are given by
# 
# \begin{align}
# &(y - x \beta)^3 / \sigma^2 \\
# &(y - x \beta)^4 / \sigma^4 - 3
# \end{align}
# 
# 
# Note: If we apply standard two-step or iterated GMM, then we would estimate the weight matrix from the empirical moment conditions without imposing further restrictions. However, GMM estimates can be noisy in small samples, and we can often improve the precision of the estimates by including available information in the calculation of the weight matrix or the covariance of the moment conditions. In this case, we could use the assumption of normality and of no heteroscedasticity also on the updating of the weight matrix (see ??? for an empirical example.) However, tests for variance that are derived under normality are in general very sensitive to deviations from normality, and we are trading of increased precision to robustness to misspecification. One example for this is the Bartlett test for the equality of variances. If we want to test the null hypothesis of normality, then imposing normality on the weight matrix would be in the spirit of score or Lagrange multiplier tests, where we want to derive our test statistic under the null. To emphasize again, these versions of test for overidentifying restrictions and LM tests are asymptotically equivalent but will differ in small sample properties.
# 
# (Related: articles that show that GMM is not very reliable and precise in small samples, especially when including variance and higher order terms. where and which references???)
# 
# **Using parameter restriction**
# 
# We can transform the previous example to testing normality as as LM test with parameter restriction. We estimate two new parameters, skew $c_1$ and excess kurtosis $c2$. The null hypothesis of normality is then that both coefficient are zero, i.e. $H0: c_1=0, \hspace{5pt} c_2=0$
# 
# \begin{align}
# &x (y - x \beta) \\
# &(y - x \beta)^2 - \sigma^2 \\
# &(y - x \beta)^3 / \sigma^2 - c_1 \\
# &(y - x \beta)^4 / \sigma^4 - 3 - c_2
# \end{align}
# 
# Because we are now in the exactly identified case, we can use the simplified version of score test $LM_{3n}$.

# In[1]:




# In[1]:




# In[1]:




# In[1]:




# The next is completely unrelated, integer chunk iterator for scipy by Evgeni

# In[2]:

import numpy as np
def _iter_chunked(x0, x1, chunksize=4, inc=1):
    """Iterate from x0 to x1 *inclusive* in chunks of chunksize and steps inc.
    x0 must be finite, x1 need not be. In the latter case, the iterator is infinite.
    Handles both x0 < x1 and x0 > x1 (in which case, iterates downwards.)
    """
    x = x0
    while (x - x1) * inc < 0:
        delta = min(chunksize, abs(x - x1))
        step = delta * inc
        supp = np.arange(x, x + step, inc)
        x += step
        yield supp


# In[2]:




# In[3]:

x0 = 0; x1 = 10
for ii in _iter_chunked(x0, x1, chunksize=3, inc=1): print(ii)


# In[4]:

x0 = 10; x1 = -1
for ii in _iter_chunked(x0, x1, chunksize=3, inc=-1): print('->', repr(ii))

