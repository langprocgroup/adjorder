"""
1. reads in GloVe vectors and a tab-delimited list of triples
  NOUN ADJ ADJ COUNT
2. counts how many of the triples arranged such that most cosine-similar adj is closest to noun
"""

import sys, codecs, numpy, pickle, os, re
#import pandas as pd
#from scipy.stats import entropy
#from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
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
            ls = line.split()
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
            if sr[0] in nouns or sr[0] in adjs: #only store adj and noun vectors
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

    if vectors_pickle_file in os.listdir():
        print("loading vector data from " + vectors_pickle_file + " ...")
        [df, labels_array] = load_from_pickle(vectors_pickle_file, [df, labels_array])
    else:
        print("building word vectors from " + args.embeddings + " ...")
        df, labels_array = build_word_vector_matrix(args.embeddings, nouns, adjs)
        print("\nsaving vector data to " + vectors_pickle_file + " ...")
        save_to_pickle(vectors_pickle_file, [df, labels_array])

    print("calculating cosine distances ...")
    correct = 0
    incorrect = 0
    tie = 0
    for i, triple in enumerate(triples):
        print_progress(i, len(triples))
        try:
            adj1 = df[labels_array.index(triple[0])]
            adj2 = df[labels_array.index(triple[1])]
            noun = df[labels_array.index(triple[2])]
            dist1 = cosine(adj1, noun)
            dist2 = cosine(adj2, noun)

            if dist1 > dist2:
                correct += 1
            elif dist1 < dist2:
                incorrect += 1
            else:
                tie += 1
        except:
            pass
        

    print("\n\n*** counts ***")
    print("   adjs: " + str(len(adjs)))
    print("  nouns: " + str(len(nouns)))
    print("triples: " + str(len(triples)))

    print("\n*** results ***")
    total = incorrect + correct
    print(str(correct/total) + " (" + str(correct) + "/" + str(total) + ") ties: " + str(tie))
    
