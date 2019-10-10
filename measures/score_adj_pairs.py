import sys, codecs, pickle, os, math
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from collections import Counter
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

def load_pairs(filename):
    pairs = pd.read_csv(filename, sep="\t", header=None)
    pairs.columns = ['count', 'adj', 'noun']
    return pairs

def load_triples(filename):
    triples = []
    with open(filename) as f:
        for line in f:
            triples.append(line.split())
    return triples

def load_vectors(filename):
    df = pd.read_csv(filename, sep='["]* ["]*', header=None, error_bad_lines=False, engine='python')
    return df

def calc_pmi(pa, pb, pab):
    return math.log(pab/float(pa*pb),2)

def ic(df, k, nouns, adjs, adj_to_nouns):
    print(" -kmeans")
    kmeans_model = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans_model.fit(df)
    labels = kmeans_model.labels_

    #build up dict to map nouns to their cluster id
    print(" -noun_clusters")
    noun_clusters = {}
    n = len(nouns)
    for i, noun in enumerate(nouns):
        try:
            noun_clusters[noun] = labels[nouns.index(noun)]
        except:
            pass
    
    # calculate each adj's entropy
    print(" -adj_ents")
    adj_ents = {}
    n = len(adjs)
    for i, adj in enumerate(adjs):
        dist = []
        try:
            nouns = adj_to_nouns[adj]
            for noun in nouns:
                try:
                    dist.append(noun_clusters[noun])
                except:
                    pass
            prob = list(Counter(dist).values())
            adj_ents[adj] = entropy(prob, base=2)
        except:
            pass

    return adj_ents

def extract_vectors(vectors, pairs):
    nv = pd.merge(vectors, pd.DataFrame(pairs.noun.unique()), how='inner', left_on=[0], right_on=[0])
    noun_vectors = np.array(nv[nv.columns[1:]].values).astype(float)
    nouns = list(nv[0].values)
    
    av = pd.merge(vectors, pd.DataFrame(pairs.adj.unique()), how='inner', left_on=[0], right_on=[0])
    adj_vectors = np.array(av[av.columns[1:]].values).astype(float)
    adjs = list(av[0].values)

    return noun_vectors, adj_vectors, nouns, adjs    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score adj pairs')
    parser.add_argument('vectors', type=str, help='GloVe file containing word vectors')
    parser.add_argument('pairs', type=str, help='file containing [count adj noun] pairs')
    parser.add_argument('triples', type=str, help='file containing [count adj adj noun] triples')
    args = parser.parse_args()

    print("loading triples data from " + args.triples + " ...")
    triples = load_triples(args.triples)
    trip_pairs = []
    for row in triples:
        trip_pairs.append([row[1], row[3]])
        trip_pairs.append([row[2], row[3]])
    trip_pairs = pd.DataFrame(trip_pairs, columns=['adj', 'noun'])

    print("loading pairs data from " + args.pairs + " ...")
    pairs = load_pairs(args.pairs)

    print("loading vectors from " + args.vectors + " ...")
    vectors = load_vectors(args.vectors)
    print("extracting noun and adj vectors ...")
    noun_vectors, adj_vectors, nouns, adjs = extract_vectors(vectors, trip_pairs)

    print("calculating probabilities ...")
    print(" -adjs")
    adj_counts = pd.pivot_table(pairs, index=['adj'], values=['count', 'adj'], aggfunc=np.sum)
    adj_probs = adj_counts.divide(np.sum(adj_counts['count'])).to_dict()['count']

    print(" -nouns")
    noun_counts = pd.pivot_table(pairs, index=['noun'], values=['count', 'noun'], aggfunc=np.sum)
    noun_probs = noun_counts.divide(np.sum(noun_counts['count'])).to_dict()['count']

    print(" -pairs")
    df = pairs.copy()
    df['pair'] = pairs['adj'] + ":" + pairs['noun']
    del df['adj']
    del df['noun']
    pair_counts = pd.pivot_table(df, index=['pair'], values=['count','pair'], aggfunc=np.sum)
    pair_probs = pair_counts.divide(np.sum(pair_counts['count'])).to_dict()['count']

    print("calculating entropies ...")
    pairs.reindex(pairs.index.repeat(pairs['count']))
    adj_to_nouns = {k: g["noun"].tolist() for k,g in pairs.groupby('adj')}
    adj_ents = ic(noun_vectors, 200, nouns, adjs, adj_to_nouns)
    
    print("printing output ...")
    outfile = open("results.csv", 'w')
    outfile.write("id,idx,count,adj,noun,adj_prob,vdist,pmi,ic\n")
    n = len(triples)
    for i, triple in enumerate(triples):
        print_progress(i+1, n)
        noun = triple[3]
        for j, adj in enumerate(triple[1:3]):
            adj_prob = None
            vdist = None
            pmi = None
            ic = None
            
            try:
                adj_prob = adj_probs[adj]
            except:
                pass
            try:
                vdist = cosine(adj_vectors[adjs.index(adj)], noun_vectors[nouns.index(noun)])
            except:
                pass
            try:
                pmi = calc_pmi(float(adj_probs[adj]), float(noun_probs[noun]), float(pair_probs[adj + ":" + noun]))
            except:
                pass
            try:
                ic = adj_ents[adj]
            except:
                pass

            outfile.write(str(i) + "," + str(j) + "," + str(triple[0]) + "," + adj + "," + noun + "," + str(adj_prob) + "," + str(vdist) + "," + str(pmi) + "," + str(ic) + "\n")
                
    outfile.close()
