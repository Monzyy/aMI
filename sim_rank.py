import networkx as net
import matplotlib.pyplot as plt


# create graph
graph = net.DiGraph()

graph.add_edge('Univ', 'ProfA')
graph.add_edge('Univ', 'ProfB')
graph.add_edge('ProfA', 'StudentA')
graph.add_edge('StudentA', 'Univ')
graph.add_edge('ProfB', 'StudentB')
graph.add_edge('StudentB', 'ProfB')


show_graph = True
if show_graph:
    net.draw(graph, pos=net.circular_layout(graph), with_labels=True)
    plt.show()

C = 0.8


def similarity_dict(g, k):

    def sim(a, b):
        sum = 0
        if a == b:
            return 1
        for a_inc, _ in g.in_edges(a):
            for b_inc, _ in g.in_edges(b):
                s = r_prev.get((a_inc, b_inc), 0.)
                sum += s
        return C / (len(g.in_edges(a)) * len(g.in_edges(b))) * sum
    r = {}
    for node in g.nodes:
        r[(node, node)] = 1
    while k > 0:
        r_prev = r.copy()
        for node1 in g.nodes:
            for node2 in g.nodes:
                r[(node1, node2)] = sim(node1, node2)
        k = k-1
    return r


def similarity(g, a, b, k):
    return similarity_dict(g, k)[(a, b)]


print(similarity(graph, 'ProfA', 'ProfB', 10))
print(similarity_dict(graph, 10))

C1 = 0.5

bigraph = net.DiGraph()
bigraph.add_edge('A', 'sugar')
bigraph.add_edge('A', 'frosting')
bigraph.add_edge('A', 'eggs')
bigraph.add_edge('B', 'frosting')
bigraph.add_edge('B', 'eggs')
bigraph.add_edge('B', 'flour')



