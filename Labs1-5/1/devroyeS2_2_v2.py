import pymc as pm
import numpy as np

c = 10
# X = c
X = pm.Uniform("X", lower = 0, upper = 4 * c)

@pm.stochastic(dtype=int)
def Y(value = 0, X = X):

    def logp(value, X):
        if value == 1:
            return np.log(X / (c + X))
        else:
            if value == 0:
                return np.log(c / (c + X))
            else:
                return -np.inf
    
    def random(X):
        return 1 if np.random.random() < (X / (c + X)) else 0

@pm.deterministic
def Yh(X = X):
    return 1 if X / (c + X) > 0.5 else 0

model = pm.Model([X, Y, Yh])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)
Y_samples = mcmc.trace('Y')[:]
Yh_samples = mcmc.trace('Yh')[:]

print()
print()
print("Bayes Error:", (Yh_samples != Y_samples).mean())



