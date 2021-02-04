import pymc as pm
from matplotlib import pyplot as plt


X1 = pm.Uniform("X1", lower = 0, upper = 1)
X2 = pm.Uniform("X2", lower = 0, upper = 1)

@pm.deterministic
def Z(X1 = X1, X2 = X2):
    return abs(X1 - X2)


model = pm.Model([X1, X2, Z])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)
Z_samples = mcmc.trace('Z')[:]

plt.hist(Z_samples, bins = 40)
plt.show()



