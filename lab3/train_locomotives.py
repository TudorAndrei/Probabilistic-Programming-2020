import pymc as pm
import matplotlib.pyplot as plt


true_L = 500

# Exponentiala da nr de firme
# Exponentiala invers da nr de locomotive

L = pm.rdiscrete_uniform(1, true_L, size=10)

N = pm.Binomial("N", n=L,)

obs = pm.DiscreteUniform("obs", lower=10, value=L, observed=True)

model = pm.Model([obs, N])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

N_samples = mcmc.trace('N')[:]

# histogram of the samples:

plt.hist(N_samples, density=True)
plt.show()
