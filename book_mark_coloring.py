import csv
import networkx as net
import numpy as np
import matplotlib.pyplot as plt

friend_graph = net.Graph()
location_graph = net.Graph()

# Load Brightkite friend graph
use_real_data = False
if use_real_data:
    with open('data/loc-brightkite_edges.txt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            friend_graph.add_edge(int(row[0]), int(row[1]))
else:
    friend_graph.add_edge(0, 2)
    friend_graph.add_edge(2, 1)

    location_graph = friend_graph.copy()

    location_graph.add_edge(0, 'l1', weight=7)
    location_graph.add_edge(0, 'l2', weight=3)
    location_graph.add_edge(2, 'l2', weight=1)
    location_graph.add_edge(2, 'l3', weight=1)

print(friend_graph.number_of_nodes(), ' nodes ', friend_graph.number_of_edges(), ' edges')

show_graph = True
if show_graph:
    net.draw(friend_graph, pos=net.spring_layout(friend_graph), with_labels=True)
    plt.show()


def bca(graph, u, alpha, epsilon):
    friends_importants = [0.] * graph.number_of_nodes()
    b = [0.] * graph.number_of_nodes()
    b[u] = 1.
    old_b = None
    while b != old_b:
        old_b = b.copy()
        for node in graph.nodes:
            if not b[node] < epsilon:
                friends_importants[node] += (1 - alpha) * b[node]
                for neighbor in graph.neighbors(node):
                    b[neighbor] += (alpha * b[node]) / len(list(graph.neighbors(node)))
                b[node] = (1 - alpha) * b[node]
    print('b:', b)
    return friends_importants


test = bca(friend_graph, 0, 0.9, 0.0001)
print(np.sum(test))
print('pi:', test)


def fbca(graph, u, d, n):
    # step 1
    visited = [neigh for neigh in graph.neighbors(u) if isinstance(neigh, str)]

    # step 2
    friend_graph = location_graph.subgraph([node for node in graph.nodes if isinstance(node, int)])
    ppr = bca(friend_graph, u, 0.9, 0.0001)

    # step 3
    location_nodes = [node for node in graph.nodes if isinstance(node, str)]
    s = {}
    not_visited = [node for node in location_graph if node not in visited]
    for node in not_visited:
            s[node] = 0

    # step 4
    for node in friend_graph:
        if node != u:
            # step 5
            for location in [neigh for neigh in graph.neighbors(node) if isinstance(neigh, str)]:
                # step 6
                s[location] = s.get(location, 0) + ppr[node] * graph.get_edge_data(node, location)['weight']
    # step 7 - 14
    res = [(k, v) for k, v in s.items()]
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:n]


print(fbca(location_graph, u=1, d=0, n=2))
