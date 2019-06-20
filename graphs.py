# Lecture 2: Constructing Graphs for Recommender Systems

import csv as csv
import networkx as net
import matplotlib.pyplot as plt


class DBClass:
    def __init__(self, name, path, pki=None, foreign_keys=None, attributes=None, delimiter='|'):
        self.name = name
        self.path = path
        self.delimiter = delimiter
        self.pki = pki
        if foreign_keys is None:
            self.foreign_keys = {}
        self.foreign_keys = foreign_keys
        if attributes is None:
            self.attributes = {}
        self.attributes = attributes

    def load_into_graph(self, gs):
        file = open(self.path, 'r', encoding='ISO-8859-1')
        reader = csv.reader(file, delimiter=self.delimiter,)
        for index, row in enumerate(file):
            if index > 30:
                break
            row = row.split(self.delimiter)
            id = self if self.pki is None else row[self.pki]
            kwargs = self.attributes
            kwargs['type'] = self.name
            gs.add_node(id, **kwargs)
            if self.foreign_keys is not None:
                for fk in self.foreign_keys.values():
                    gs.add_edge(id, int(row[fk]))


def load_data_complex(G):
    dir = 'data/ml-100k/'

    # define user class
    user_attributes = {'age': 1, 'gender': 2, 'occupation': 3, 'created_at': 4}
    user_class = DBClass('user', dir + 'u.user', pki=0, attributes=user_attributes)

    # define movie class
    movie_attributes = {'title': 1, 'release_date': 2, 'video_release_date': 3, 'IMDb_url': 4}
    movie_class = DBClass('movie', dir + 'u.item', pki=0, attributes=movie_attributes)

    # define rating class
    rating_attributes = {'rating': 2, 'timestamp': 3}
    rating_foreign_keys = {'user': 0, 'movie': 1}
    rating_class = DBClass('rating', dir + 'u.data', attributes=rating_attributes,
                           foreign_keys=rating_foreign_keys, delimiter='\t')

    # load data into graph
    user_class.load_into_graph(G)
    movie_class.load_into_graph(G)
    rating_class.load_into_graph(G)


def load_data_easy(G, load_subset=False):
    file = open('data/ml-100k/u1.base', 'r')
    moviereader = csv.reader(file, delimiter='\t')

    for index, row in enumerate(moviereader):
        if load_subset and index > 100:
            break
        # print(', '.join(row))
        # according to readme the format of the row is userid, movieid, rating, timestamp
        # creating graph with nodes for each item, edges between user and item are through rating (substituting play from the
        # paper) and timestamp is connected to play
        # userid
        G.add_node((row[0], 'user'))
        # itemid
        G.add_node((row[1], 'item'))
        # rating/playing
        G.add_node((row[0], row[1]))
        # user - rating
        G.add_edge((row[0], 'user'), (row[0], row[1]))
        # item - rating
        G.add_edge((row[1], 'item'), (row[0], row[1]))


def personalized_teleport_vector(user_id, graph: net.Graph):
    q = {}
    node = (str(user_id), 'user')
    neighbors = graph[node]
    n_neighbors = len(neighbors.items())

    for neighbor in neighbors:
        q[neighbor] = 1/n_neighbors
    return q


def top_k_recommendations(user_id, graph, k=10):
    q = personalized_teleport_vector(user_id, graph)
    pr = net.pagerank(graph, personalization=q)
    already_seen_items = []
    user_node = (str(user_id), 'user')
    neighbors = graph[user_node]
    for neighbor in neighbors:
        already_seen_items.append(neighbor[1])
    pr = {k: v for k, v in pr.items() if k[0] not in already_seen_items and k[1] == 'item'}
    return sorted(pr.items(), reverse=True, key=lambda kv: kv[1])[:k]


def top_k_hitrate(user_id, graph, k=10):
    top_k = top_k_recommendations(user_id, graph, k)
    file = open('data/ml-100k/u1.test', 'r')
    test_reader = csv.reader(file, delimiter='\t')
    rated_test_items = []
    for line in test_reader:
        if line[0] == str(user_id):
            rated_test_items.append(line[1])
    hits = 0
    for rating in top_k:
        if rating[0][0] in rated_test_items:
            hits += 1

    if len(rated_test_items) < k:
        k = len(rated_test_items)
    if k == 0:
        return 0
    return hits/k


def get_all_users():
    res = set()
    with open('data/ml-100k/u1.base', 'r') as file:
        moviereader = csv.reader(file, delimiter='\t')
        for row in moviereader:
            res.add(row[0])
    return res


def average_top_k_hitrate(graph, k=10):
    all_users = get_all_users()
    sum = 0
    for user in all_users:
        sum = top_k_hitrate(user, graph, k)
    return sum/len(all_users)


G = net.Graph()

ez_loading = True

if ez_loading:
    load_data_easy(G)
else:
    load_data_complex(G)

for node in G.nodes:
    neighbors = G[node]
    n_neighbors = len(neighbors.items())
    for neighbor in neighbors:
        G.add_edge(node, neighbor, weight=1/n_neighbors)


user_id = 1
print(top_k_hitrate(1, G))
#print(average_top_k_hitrate(G, k=10))
recs = top_k_recommendations(user_id, G)
for k, v in recs:
    print(k, v)

# ### Building aggregated graph
# Aggregating on play - item edges so there's only one weighted edge between an item a play indicating number of plays
AG = net.Graph()

items = [node for node in G.nodes if node[1] == 'item']
users = [node for node in G.nodes if node[1] == 'user']
ratings = [node for node in G.nodes if node[1] != 'item' and node[1] != 'user']


for item in items:
    AG.add_node(item)
    edges = G[item]
    rating = ('r', item[0])
    AG.add_node(rating)
    AG.add_weighted_edges_from([(item, rating, len(edges))])

for user in users:
    AG.add_node(user)

for rating in ratings:
    for user in users:
        if G.has_edge(user, rating):
            AG.add_edge(user, ('r', rating[1]))

print('G: ', G.number_of_edges(), ' edges ', G.number_of_nodes(), ' nodes')
print('AG: ', AG.number_of_edges(), ' edges ', AG.number_of_nodes(), ' nodes')
print(top_k_hitrate(user_id, AG))
recs = top_k_recommendations(user_id, AG)
for k, v in recs:
    print(k, v)
print(average_top_k_hitrate(AG, k=3))

#q = personalization_vector(user_id, G)
#pr = net.pagerank(G, personalization=q)
#sorted_pr = sorted(pr.items(), reverse=True, key=lambda kv: kv[1])


# subgraph (instead of NE-aggregation) function in networkx
# disjoint_union as merge function


#net.draw(G, pos=net.kamada_kawai_layout(G))
#plt.show()
