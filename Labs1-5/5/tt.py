import networkx as nx
from matplotlib import pyplot as plt

G = nx.waxman_graph(10, alpha=0.5, beta=0.1, L=9, domain=(1, 0, 10, 0))

nx.draw(G, with_labels=True, font_weight='bold')

plt.show()
