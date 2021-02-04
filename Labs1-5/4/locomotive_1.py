import numpy as np
import pymc as pm
from matplotlib import pyplot as plt


true_N = 500
D = pm.rdiscrete_uniform(1, true_N, size = 10)

#N = pm.DiscreteUniform("N", lower=D.max(), upper=10000)

alpha = 1
Ns = np.arange(D.max(), 100001)
Ns = Ns ** (-alpha)
Ns = Ns / np.sum(Ns)

N_ = pm.Categorical("N", Ns)

@pm.deterministic(dtype = "int")
def N(N_ = N_, shift = D.max()):
   return shift + N_

observation = pm.DiscreteUniform("obs", lower=0, upper=N, value=D, observed=True)

model = pm.Model([observation, N])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

N_samples = mcmc.trace('N')[:]

# histogram of the samples:

plt.hist(N_samples)
plt.show()
