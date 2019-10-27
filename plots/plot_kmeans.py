import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess, argparse, sys
from sklearn.cluster import KMeans
from collections import Counter

def load_file(filename):
    df = pd.read_csv(filename, sep=",")
    return df

def load_pairs(filename):
    pairs = pd.read_csv(filename, sep="\t", header=None)
    pairs.columns = ['count', 'awf', 'nwf']
    pairs['awf'] = pairs['awf'].str.lower()
    pairs['nwf'] = pairs['nwf'].str.lower()
    return pairs

def load_vectors(filename):
    df = pd.read_csv(filename, sep='["]* ["]*', header=None, error_bad_lines=False, engine='python')
    return df

def extract_vectors(vectors, pairs):
    nv = pd.merge(vectors, pd.DataFrame(pairs.nwf.unique()), how='inner', left_on=[0], right_on=[0])
    noun_vectors = np.array(nv[nv.columns[1:]].values).astype(float)

    av = pd.merge(vectors, pd.DataFrame(pairs.awf.unique()), how='inner', left_on=[0], right_on=[0])
    adj_vectors = np.array(av[av.columns[1:]].values).astype(float)

    return noun_vectors, adj_vectors

def count_clusters(vectors, k):
    kmeans_model = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans_model.fit(vectors)
    labels = kmeans_model.labels_

    counts = Counter(labels)

    return Counter(counts.values())[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot kmeans')
    parser.add_argument('vectors', type=str, help='GloVe file containing word vectors')
    parser.add_argument('pairs', type=str, help='file containing [count adj noun] pairs')
    args = parser.parse_args()

    print("loading pairs data from " + args.pairs + " ...")
    pairs = load_pairs(args.pairs)

    print("loading vectors from " + args.vectors + " ...")
    vectors = load_vectors(args.vectors)

    print("extracting noun and adj vectors ...")
    nwf_vectors, awf_vectors = extract_vectors(vectors, pairs)

    n_adj = len(awf_vectors)
    n_noun = len(nwf_vectors)

    nks = 10
    xmin = 50
    adj_xmax = int(n_adj/10)
    noun_xmax = int(n_noun/10)
    adj_xstep = int(adj_xmax/nks)
    noun_xstep = int(noun_xmax/nks)
    adj_singles = []
    noun_singles = []

    #adj_singles = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 2, 6, 3, 9, 12, 17, 13]
    #noun_singles = [0, 0, 0, 0, 0, 1, 1, 0, 2, 4, 8, 12, 16, 20, 29, 35, 46, 52, 68, 87]


    for k in range(xmin,adj_xmax,adj_xstep):
        print(str(k) + "-means adjs ...")
        adj_singles.append(count_clusters(awf_vectors, k))

    for k in range(xmin, noun_xmax, noun_xstep):
        print(str(k) + "-means nouns ...")
        noun_singles.append(count_clusters(nwf_vectors, k))

    print(adj_singles)
    print(noun_singles)

    adj_df = pd.DataFrame({'x': range(xmin,adj_xmax,adj_xstep), 'singles': adj_singles})
    noun_df = pd.DataFrame({'x': range(xmin,noun_xmax,noun_xstep), 'singles': noun_singles})    

    plt.figure(1,figsize=(12,6))
    
    plt.subplot(1, 2, 1)
    #plt.plot('x', 'x', data=adj_df, linestyle='-', color='blue', label='# clusters')
    plt.plot('x', 'singles', data=adj_df, linestyle='-', color='green', label='singletons')
    #plt.axhline(len(awf_vectors), linestyle='--', label='# wordforms')
    plt.title("adjectives (n=" + str(n_adj) + ")")
    plt.xlabel("k")
    plt.ylabel("# singleton clusters")
    #plt.legend()
    

    plt.subplot(1, 2, 2)
    #plt.plot('x', 'count', data=noun_df, linestyle='-', color='blue', label='# clusters')
    plt.plot('x', 'singles', data=noun_df, linestyle='-', color='green', label='singletons')
    #plt.axhline(len(nwf_vectors), linestyle='--', label='# wordforms')    
    plt.title("nouns (n=" + str(n_noun) + ")")
    plt.xlabel("k")
    #plt.ylabel("rate")
    #plt.legend()

    plt.tight_layout()
    plt.savefig("plot_kmeans.png", bbox_inches='tight', pad_inches=0)
