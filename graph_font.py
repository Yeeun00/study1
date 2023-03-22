import networkx as nx
import matplotlib.pyplot asG = nx.Graph()

plt
plt.rc('font', family='NanumBarunGothicOTF')

G = nx.Graph()

node_data = ['정','박','이']
G.add_nodes_from(node_data)

edge_data = [('정','박'),('정','이'),('박','이'),('정','박')]
G.add_edges_from(edge_data)

#pos = nx.spring_layout(G)
#nx.draw(G,pos, with_labels=True)

pos=nx.shell_layout(G)
nx.draw(G)
nx.draw_networkx_labels(G, pos, font_family='AppleGothic', font_size=10)

plt.show()
