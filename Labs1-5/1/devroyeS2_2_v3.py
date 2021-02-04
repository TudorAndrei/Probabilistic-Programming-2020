import pymc as pm

c = 10
# X = c
X = pm.Uniform("X", lower = 0, upper = 4 * c)

@pm.deterministic
def loss(X = X):
    return c / (c + X) if c < X else X / (c + X)

model = pm.Model([X, loss])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)
loss_samples = mcmc.trace('loss')[:]

print()
print()
print("Bayes Error:", loss_samples.mean())


