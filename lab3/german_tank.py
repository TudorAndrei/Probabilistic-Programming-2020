import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

true_N = 500
number_of_tanks = pm.rdiscrete_uniform(1, true_N, size=10)

N = pm.DiscreteUniform("N", lower=number_of_tanks.max(), upper=10000)

observation = pm.DiscreteUniform(
    "obs", lower=0, upper=N, value=number_of_tanks, observed=True)

model = pm.Model([observation, N])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

N_samples = mcmc.trace('N')[:]

# histogram of the samples:

plt.hist(N_samples, density=True)
plt.avline(np.mean(N_samples))
plt.show()
