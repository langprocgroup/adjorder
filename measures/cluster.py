import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

def print_progress(i, n):
    j = (i+1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% N %d/%d " % ('='*int(20*j), 100*j, i, n))
    sys.stdout.flush()
    return i + 1

def load_pairs(filename):
    pairs = pd.read_csv(filename, sep=",", header=None)
    pairs.columns = ['count', 'awf', 'nwf']
    pairs['awf'] = pairs['awf'].str.lower()
    pairs['nwf'] = pairs['nwf'].str.lower()
    return pairs

def load_vectors(filename, c):
    df = pd.read_csv(filename, sep='["]* ["]*', header=None, error_bad_lines=False, engine='python', index_col=0)
    if c != 1.0:
        print("running pca(" + str(c) + ") ...")
        x = run_pca(df.values, c)
        w = df.index
        df = pd.concat([pd.DataFrame(w), pd.DataFrame(x)], axis=1, ignore_index=True)
        df.set_index([0], inplace=True)
    return df

def run_pca(x, c):
    xdim = x.shape[1]
    pca = PCA(c)
    x = StandardScaler().fit_transform(x)
    principalComponents = pca.fit_transform(x)
    print("  dim: " + str(xdim) + " -> " + str(pca.n_components_))
    return principalComponents

def get_clusters(words, vectors, k):
    kmeans_model = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans_model.fit(vectors)
    labels = kmeans_model.labels_

    clusters = {}
    for i, word in enumerate(words):
        clusters[word] = int(labels[i])

    return clusters
        
def extract_vectors(vectors, pairs):
    nv = pd.merge(pd.DataFrame(pairs.nwf.unique()), vectors, left_on=[0], right_index=True)
    noun_vectors = np.array(nv[nv.columns[1:]].values).astype(float)
    nouns = list(nv[0].values)

    av = pd.merge(pd.DataFrame(pairs.awf.unique()), vectors, left_on=[0], right_index=True)
    adj_vectors = np.array(av[av.columns[1:]].values).astype(float)
    adjs = list(av[0].values)

    return noun_vectors, adj_vectors, nouns, adjs    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score adj pairs')
    parser.add_argument('-v', '--vectors', nargs=1, dest='vectors', required=True, help='GloVe file containing word vectors')
    parser.add_argument('-p', '--pairs', nargs=1, dest='pairs', required=True, help='file containing [count adj noun] pairs')
    parser.add_argument('-k', nargs=1, dest='k', required=False, help='pair of k values for clustering; default is 300,1000')
    parser.add_argument('-c', nargs=1, dest='c', type=float, required=False, help='pct of components to keep after pca; default is 1.0 (all)')
    args = parser.parse_args()

    try:
        k = args.k[0].trim().split(",")
        ak = k[0]
        nk = k[1]
    except:
        ak = 300
        nk = 1000

    try:
        c = args.c[0]
    except:
        c = 1.0

    print("loading pairs data from " + args.pairs[0] + " ...")
    pairs = load_pairs(args.pairs[0])

    print("loading vectors from " + args.vectors[0] + " ...")
    vectors = load_vectors(args.vectors[0], c)

    print("extracting noun and adj vectors ...")
    nwf_vectors, awf_vectors, nwfs, awfs = extract_vectors(vectors, pairs)

    print("clustering adjs into " + str(ak) + " clusters ...")
    acls = get_clusters(awfs, awf_vectors, ak)
    pairs['acl'] = pairs['awf'].map(acls).astype('str')

    print("clustering nouns into " + str(nk) + " clusters ...")
    ncls = get_clusters(nwfs, nwf_vectors, nk)
    pairs['ncl'] = pairs['nwf'].map(ncls).astype('str')

    print("printing output to clust_pairs.csv ...")
    outfile = open("clust_pairs.csv", 'w')
    outfile.write(pairs.to_csv(columns=['count', 'awf', 'nwf', 'acl', 'ncl'], index=False))
    outfile.close()
    
