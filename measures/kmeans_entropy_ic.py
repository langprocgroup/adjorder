"""
1. reads in GloVe vectors and a tab-delimited list of triples
  NOUN ADJ ADJ COUNT
2. generates k-means clusters of the noun vectors
3. finds the prob dist of clusters for each adj and takes entropy of that dist
4. counts how many of the triples have high-entropy adj first
"""

import sys, codecs, numpy, pickle, os, re
import pandas as pd
from scipy.stats import entropy
from collections import Counter
from sklearn.cluster import KMeans

import subprocess, argparse

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def print_progress(i, n):
    j = (i+1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% N %d/%d " % ('='*int(20*j), 100*j, i, n))
    sys.stdout.flush()
    return i + 1

def load_data(filename):
    pairs = []
    triples = []
    n = file_len(filename)
    i = 1
    with open(filename) as f:
        for line in f:
            i = print_progress(i, n)
            ls = re.sub('/[A-Z]*', '', line).split()
            for j in range(int(ls[3])):
                pairs.append([ls[1], ls[0]])
                pairs.append([ls[2], ls[0]])
            triples.append([ls[1], ls[2], ls[0]])
    return pairs, triples

def build_word_vector_matrix(vector_file, nouns, adjs):
    numpy_arrays = []
    labels_array = []
    n = file_len(vector_file)
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, r in enumerate(f):
            print_progress(i+1, n)
            sr = r.split()
            if sr[0] in nouns or sr[0] in adjs: #we only care about nouns here, but other methods need adjs, so pickle data should have both
                labels_array.append(sr[0])
                numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )
        
    return numpy.array( numpy_arrays ), labels_array

def load_from_pickle(filename, objects):
    f = open(filename, 'rb')
    return_objects = []
    for obj in objects:
        return_objects.append(pickle.load(f))
    return return_objects

def save_to_pickle(filename, objects):
    f = open(filename, 'wb')
    for obj in objects:
        pickle.dump(obj, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='integration cost')
    parser.add_argument('embeddings', type=str, help='GloVe file containing word vectors')
    parser.add_argument('triples', type=str, help='file containing [count adj adj noun] triples')    
    args = parser.parse_args()

    pairs = None
    triples = None
    df = None
    labels_array = None
    triples_pickle_file = "triples.pickle"
    vectors_pickle_file = "vectors.pickle"

    if triples_pickle_file in os.listdir():
        print("loading triples data from " + triples_pickle_file + " ...")
        [pairs, triples] = load_from_pickle(triples_pickle_file, [pairs, triples])
    else:
        print("loading triples data from " + args.triples + " ...")
        pairs, triples = load_data(args.triples)
        print("\nsaving triples data to " + triples_pickle_file + " ...")
        save_to_pickle(triples_pickle_file, [pairs, triples])
    adjs = set([row[0] for row in pairs])
    nouns = set([row[1] for row in pairs])
    pdf = pd.DataFrame(pairs, columns=['adj', 'noun'])
    adj_to_nouns = {k: g["noun"].tolist() for k,g in pdf.groupby("adj")}

    if vectors_pickle_file in os.listdir():
        print("loading vector data from " + vectors_pickle_file + " ...")
        [df, labels_array] = load_from_pickle(vectors_pickle_file, [df, labels_array])
    else:
        print("building word vectors from " + args.embeddings + " ...")
        df, labels_array = build_word_vector_matrix(args.embeddings, nouns, adjs)
        print("\nsaving vector data to " + vectors_pickle_file + " ...")
        save_to_pickle(vectors_pickle_file, [df, labels_array])

    print("\n*** counts ***")
    print("   adjs: " + str(len(adjs)))
    print("  nouns: " + str(len(nouns)))
    print("triples: " + str(len(triples)))

    print("\n*** results ***")
    for k in range(10,500,10):

        #cluster
        kmeans_model = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans_model.fit(df)
        labels = kmeans_model.labels_

        #build up noun_clusters dict
        noun_clusters = {}
        n = len(nouns)
        for i, noun in enumerate(nouns):
            try:
                noun_clusters[noun] = labels[labels_array.index(noun)]
            except:
                pass

        # calculate each adj's entropy
        adj_ents = {}
        n = len(adjs)
        for i, adj in enumerate(adjs):
            dist = []
            for noun in adj_to_nouns[adj]:
                try:
                    dist.append(noun_clusters[noun])
                except:
                    pass        
            prob = list(Counter(dist).values())
            adj_ents[adj] = entropy(prob, base=2)
        
        # count how many triples have high-entropy adj first
        correct = 0
        incorrect = 0
        tie = 0
        for triple in triples:
            e1 = adj_ents[triple[0]]
            e2 = adj_ents[triple[1]]
            if e1 > e2:
                correct += 1
            elif e1 < e2:
                incorrect += 1
            else:
                tie += 1
        total = incorrect + correct
        print(str(k) + "-means: " + str(correct/total) + " (" + str(correct) + "/" + str(total) + ") ties: " + str(tie))
    
