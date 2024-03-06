import networkx as nx
import matplotlib.pyplot as plt

g = nx.read_graphml("graph.ml")

plt.figure(figsize=(20,20))
pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
#pos = nx.spring_layout(g, k=10)
options = {
    "font_size": 10,
    "font_color": "white",
    "node_size": 500,
    "node_color": "blue",
    "edgecolors": "gray",
    "linewidths": 1,
    "width": 1
}
fig = nx.draw_networkx(g, pos=pos, arrows=True, **options)

#nx.draw_networkx_labels(
#    g, pos, labels=labels, font_color="green", font_size=16
#)
plt.tight_layout()
plt.axis("off")
plt.show()



