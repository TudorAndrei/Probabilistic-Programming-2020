import numpy as np
import pymc as pm
import networkx as nx
from matplotlib import pyplot as plt

data = np.load('cwg.npy')

L= 9.0

alpha = pm.Uniform('alpha', 0, 1)
beta = pm.Uniform('beta', 0, 1)

@pm.observed(dtype=nx.Graph)
def cwg(value = data, alpha = alpha, beta = beta, L = L):
    tmp = 0
    for k in range(value.size):
        for i in range(1, len(value[k])):
            for j in range(i + 1, len(value[k])+1):
                if value[k].has_edge(i, j):
                    tmp += np.log(alpha) - ((j - i) / (beta * L))
                else:
                    tmp += np.log(1 - alpha * np.exp((i - j) / (beta * L)))
    return tmp

mcmc = pm.MCMC([alpha, beta, L, cwg])
mcmc.sample(20000, 10000)

alpha_samples = mcmc.trace('alpha')[:]
beta_samples = mcmc.trace('beta')[:]


plt.hist(alpha_samples)
plt.show()

plt.hist(beta_samples)
plt.show()
