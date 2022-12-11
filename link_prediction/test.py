import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
'''

# Load the adjacency matrix into a numpy array.
a = np.loadtxt('mat_drug_drug.txt', dtype=int)

print ("a:")
print (a)

num_nodes = a.shape[0] + a.shape[1]

# Get the row and column coordinates where the array is 1.
rows, cols = np.where(a == 1)

# We label the nodes corresponding to the rows with integers from 0 to
# a.shape[0]-1, and we label the nodes corresponding to the columns with
# integers from a.shape[0] to a.shape[0] + a.shape[1] - 1.
# Rearranges the list of rows and columns into a list of edge tuples.
edges = zip(rows.tolist(), (cols + a.shape[0]).tolist())
print ("U nodes:", np.arange(a.shape[0]))
print ("V nodes:", np.arange(a.shape[1]) + a.shape[0])
print ("edges")
print (edges)

# Create a Graph object (from the networkx library).
b = nx.Graph()
b.add_nodes_from(range(num_nodes))  # This line not strictly necessry.
b.add_edges_from(edges)

# Draw the graph.  First create positions for each node. Put the U nodes
# on the left (x=1) and the V nodes on the right (x=2).
pos = dict([(k, (1, k - 0.5 * a.shape[0]))
            for k in range(a.shape[0])])
pos.update(dict([(k + a.shape[0], (2, k - 0.5 * a.shape[1]))
                  for k in range(a.shape[1])]))
nx.draw_networkx(b, pos=pos, node_color=(['c'] * a.shape[0]) + (['y'] * a.shape[1]))

plt.axis('off')
plt.show()'''
rng = np.random.default_rng()
a = rng.integers(low=0, high=2, size=(10, 10))
DG = nx.from_numpy_array(a, create_using=nx.DiGraph)

print(DG)