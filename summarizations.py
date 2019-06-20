# Lecture 1: Multi Document Summarization

import itertools
import math

import matplotlib.pyplot as plt
import networkx as net
import nltk as nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load text from semester project
file = open('data/paper.txt', mode='r')
content = file.read()
file.close()

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))
i_sentences = nltk.sent_tokenize(content)
i_words = nltk.word_tokenize(content)


# tokenizing. Each sentence becomes a token. Stemming by removing stopwords. i.e. a, an, etc. And stemming by reducing
# words to their base. For example: argue, argued, argues, arguing, argus --> argu
sentences = []
for token in i_sentences:
    stripped = ' '.join([ps.stem(w) for w in token.split() if w.lower() not in stop_words])
    if stripped:
        sentences.append(stripped)


# ### Similarity based on co-occurences ### #
def coocurrence_similarity(sent_vec1, sent_vec2):
    overlap = len(set(sent_vec1.split()).intersection(set(sent_vec2.split())))
    return overlap / (math.log10(len(sent_vec1)) + math.log10(len(sent_vec2)))


def row_normalize(array):
    return array / array.sum(axis=0)


n_sentences = len(sentences)
similiarity_coocurrence = np.zeros(shape=(n_sentences, n_sentences))
# generating edges with value of co-occurences similarity
for idx, x in enumerate(sentences):
    for idy, y in enumerate(sentences):
        similiarity_coocurrence[idx, idy] = coocurrence_similarity(x, y) if idx != idy else 0


normalized_similarities = row_normalize(similiarity_coocurrence)

GS = net.Graph()
for idx, x in enumerate(sentences):
    for idy, y in enumerate(sentences):
        GS.add_edge(x, y, weight=similiarity_coocurrence[idx, idy])

# Draw coocurrence graph
net.draw(GS, pos=net.spring_layout(GS), label='Co-ocurrence graph')
plt.show()



# ### Calculate pagerank ### #
pagerank = net.pagerank(GS, alpha=0.85)
# alpha: Damping parameter there's a 1-alpha chance to go to a random node
print(pagerank)


# ### Cosine similarity with tfidf vectorization ### #
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(sentences)
cos_sim_matrix = cosine_similarity(tfidf)

graph_cos_sim = net.Graph()
for idx, x in enumerate(sentences):
    for idy, y in enumerate(sentences):
        graph_cos_sim.add_edge(x, y, weight=cos_sim_matrix[idx, idy])

net.draw(graph_cos_sim, pos=net.spring_layout(graph_cos_sim))


# I used my semester project paper as document.
# Cosine similarity graph is not shown, but it is not very interesting.
# The first couple sentences are from the abstract, which could indicate the top-k sentences
# is a good summary of the paper
# I used python libraries so I didn't have implement the algorithms from scratch
