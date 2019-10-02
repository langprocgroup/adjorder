import sys
import pandas as pd
from gensim.models import Word2Vec
from sklearn import cluster
from scipy.stats import entropy
from collections import Counter

def load_data(filename):
    pairs = []
    triples = []
    with open(filename) as f:
        for line in f:
            ls = line.split()
            for i in range(int(ls[0])):
                pairs.append([ls[1], ls[3]])
                pairs.append([ls[2], ls[3]])
                triples.append([ls[1], ls[2], ls[3]])
    return pairs, triples

def get_word_vectors(pairs):
    model = Word2Vec(pairs, min_count=1)
    X = model.wv
    del model
    return X

def do_cluster(X, k):
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels

if __name__ == '__main__':
    try:
        pairs, triples = load_data(sys.argv[1])
    except:
        print("Usage: python kmeans_entropy_ic.py <file>\n  where <file> is a tab-separated file of\n  count adj adj noun")
        exit()
    adjs = [row[0] for row in pairs]
    nouns = [row[1] for row in pairs]
    pairs_df = pd.DataFrame(pairs)
    
    X = get_word_vectors(pairs)

for k in range(2,10):

    #cluster
    labels = do_cluster(X.vectors, k)

    #build up noun_clusters dict
    noun_clusters = {}
    for noun in set(nouns):
        noun_clusters[noun] = labels[X.vocab[noun].index]
    
    # calculate each adj's entropy
    adj_ents = {}
    for adj in set(adjs):
        dist = []
        for noun in pairs_df.loc[pairs_df[0] == adj][1].values:
            dist.append(noun_clusters[noun])
        prob = list(Counter(dist).values())
        adj_ents[adj] = entropy(prob, base=2)
        
    # count how many triples have high-entropy adj first
    correct = 0
    total = 0
    for triple in triples:
        e1 = adj_ents[triple[0]]
        e2 = adj_ents[triple[1]]
        if e1 > e2:
            correct += 1
        if e1 != e2:
            total += 1

    print(str(k) + "-means: " + str(correct/total) + " (" + str(correct) + "/" + str(total) + ")")
    
