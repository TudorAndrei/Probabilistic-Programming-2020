import pymc as pm
from matplotlib import pyplot as plt


N = 10000

data = pm.rnormal(2, 1000) + pm.rnormal(8, 1000)

m1 = pm.Uniform("m1", lower=1, upper=10)
m2 = pm.Uniform("m2", lower=1, upper=10)

X1 = pm.Normal("X1", m1, 100, size=N)
#X2 = pm.Uniform("X2", m2, 100, size = N)


@pm.potential
def X2(X=X1, Z=data):
    return pm.normal_like(Z - X, 8, 100)


model = pm.Model([m1, m2, X1, X2])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)
m1_samples = mcmc.trace('m1')[:]
m2_samples = mcmc.trace('m2')[:]


plt.hist(m1_samples, normed=True)
plt.show()

plt.hist(m2_samples, normed=True)
plt.show()
