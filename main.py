import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_node('A')
G.add_node('B')
G.add_edge('A', 'B', arrowstyle='<|-')
G.add_edge('B', 'A', arrowstyle='->')
nx.draw(G, with_labels=True,
        node_size=500,
        node_color="#CBCFD1",
        pos=nx.circular_layout(G))
plt.show()



