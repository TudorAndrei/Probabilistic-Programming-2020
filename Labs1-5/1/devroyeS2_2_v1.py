import pymc as pm

c = 10
# X = c
X = pm.Uniform("X", lower=0, upper=4 * c)
Y = pm.Bernoulli("Y", X / (c + X))


@pm.deterministic
def Yh(X=X):
    return 1 if X / (c + X) > 0.5 else 0


model = pm.Model([X, Y, Yh])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)
Y_samples = mcmc.trace('Y')[:]
Yh_samples = mcmc.trace('Yh')[:]

print()
print()
print("Bayes Error:", (Yh_samples != Y_samples).mean())
