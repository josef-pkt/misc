
# coding: utf-8

## GMM - Notation

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
# maxiter = 0, efficient=True :   $$(G(\theta^{(1)})' \hspace{3pt} W^{(0)} \hspace{3pt} G(\theta^{(1)}))^{-1}$    $
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

# In[ ]:



