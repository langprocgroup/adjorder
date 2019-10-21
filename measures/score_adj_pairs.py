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

def load_pairs(filename):
    pairs = pd.read_csv(filename, sep="\t", header=None)
    pairs.columns = ['count', 'awf', 'nwf']
    pairs['awf'] = pairs['awf'].str.lower()
    pairs['nwf'] = pairs['nwf'].str.lower()
    return pairs

def load_triples(filename):
    triples = []
    with open(filename) as f:
        for line in f:
            triples.append(line.lower().split())
    return triples

def load_vectors(filename):
    df = pd.read_csv(filename, sep='["]* ["]*', header=None, error_bad_lines=False, engine='python')
    return df

def calc_pmi(pa, pb, pab):
    return math.log(pab/float(pa*pb),2)

def get_clusters(words, vectors, k):
    kmeans_model = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans_model.fit(vectors)
    labels = kmeans_model.labels_

    clusters = {}
    for i, word in enumerate(words):
        clusters[word] = float(labels[i])

    return clusters

def ic(d):
    ents = {}
    for k in d:
        ents[k] = entropy(list(Counter(d[k]).values()), base=2)
    return ents
        
def extract_vectors(vectors, pairs):
    nv = pd.merge(vectors, pd.DataFrame(pairs.nwf.unique()), how='inner', left_on=[0], right_on=[0])
    noun_vectors = np.array(nv[nv.columns[1:]].values).astype(float)
    nouns = list(nv[0].values)
    
    av = pd.merge(vectors, pd.DataFrame(pairs.awf.unique()), how='inner', left_on=[0], right_on=[0])
    adj_vectors = np.array(av[av.columns[1:]].values).astype(float)
    adjs = list(av[0].values)

    return noun_vectors, adj_vectors, nouns, adjs    

if __name__ == '__main__':
    #awf = adjective wordform
    #acl = adjective cluster
    #nwf = noun wordform
    #ncl = noun cluster
    
    parser = argparse.ArgumentParser(description='score adj pairs')
    parser.add_argument('vectors', type=str, help='GloVe file containing word vectors')
    parser.add_argument('pairs', type=str, help='file containing [count adj noun] pairs')
    parser.add_argument('triples', type=str, help='file containing [count adj adj noun] triples')
    args = parser.parse_args()

    print("loading triples data from " + args.triples + " ...")
    triples = load_triples(args.triples)

    pickle_file = "data.pkl"
    pairs = None
    ncls = None
    acls = None
    nwf_vectors = None
    awf_vectors = None
    nwfs = None
    awfs = None

    if pickle_file in os.listdir():
        print("loading pairs data from " + pickle_file + " ...")
        [pairs, ncls, acls, nwf_vectors, awf_vectors, nwfs, awfs] = load_from_pickle(pickle_file, [pairs, ncls, acls, nwf_vectors, awf_vectors, nwfs, awfs])
    else:
        print("loading pairs data from " + args.pairs + " ...")
        pairs = load_pairs(args.pairs)

        print("loading vectors from " + args.vectors + " ...")
        vectors = load_vectors(args.vectors)

        print("extracting noun and adj vectors ...")
        nwf_vectors, awf_vectors, nwfs, awfs = extract_vectors(vectors, pairs)

        print("clustering adjs ...")
        acls = get_clusters(awfs, awf_vectors, 200)
        pairs['acl'] = pairs['awf'].map(acls).astype('str')

        print("clustering nouns ...")
        ncls = get_clusters(nwfs, nwf_vectors, 200)
        pairs['ncl'] = pairs['nwf'].map(ncls).astype('str')

        print("adding pairs to dataframe ...")
        pairs['awf_nwf'] = pairs['awf'] + "_" + pairs['nwf']
        pairs['awf_ncl'] = pairs['awf'] + "_" + pairs['ncl']
        pairs['acl_nwf'] = pairs['acl'] + "_" + pairs['nwf']
        pairs['acl_ncl'] = pairs['acl'] + "_" + pairs['ncl']

        print("saving data to " + pickle_file + " ...")
        save_to_pickle(pickle_file, [pairs, ncls, acls, nwf_vectors, awf_vectors, nwfs, awfs])
    
    print("calculating probabilities ...")
    print(" -awfs")
    awf_counts = pd.pivot_table(pairs[['count', 'awf']], index=['awf'], values=['count', 'awf'], aggfunc=np.sum)
    awf_probs = awf_counts.divide(np.sum(awf_counts['count'])).to_dict()['count']

    print(" -acls")
    acl_counts = pd.pivot_table(pairs[['count', 'acl']], index=['acl'], values=['count', 'acl'], aggfunc=np.sum)
    acl_probs = acl_counts.divide(np.sum(acl_counts['count'])).to_dict()['count']

    print(" -nwfs")
    nwf_counts = pd.pivot_table(pairs[['count', 'nwf']], index=['nwf'], values=['count', 'nwf'], aggfunc=np.sum)
    nwf_probs = nwf_counts.divide(np.sum(nwf_counts['count'])).to_dict()['count']

    print(" -ncls")
    ncl_counts = pd.pivot_table(pairs[['count', 'ncl']], index=['ncl'], values=['count', 'ncl'], aggfunc=np.sum)
    ncl_probs = ncl_counts.divide(np.sum(ncl_counts['count'])).to_dict()['count']

    
    print(" -awf_nwf")
    awf_nwf_counts = pd.pivot_table(pairs[['count', 'awf_nwf']], index=['awf_nwf'], values=['count', 'awf_nwf'], aggfunc=np.sum)
    awf_nwf_probs = awf_nwf_counts.divide(np.sum(awf_nwf_counts['count'])).to_dict()['count']

    print(" -awf_ncl")
    awf_ncl_counts = pd.pivot_table(pairs[['count', 'awf_ncl']], index=['awf_ncl'], values=['count', 'awf_ncl'], aggfunc=np.sum)
    awf_ncl_probs = awf_ncl_counts.divide(np.sum(awf_ncl_counts['count'])).to_dict()['count']

    print(" -acl_nwf")
    acl_nwf_counts = pd.pivot_table(pairs[['count', 'acl_nwf']], index=['acl_nwf'], values=['count', 'acl_nwf'], aggfunc=np.sum)
    acl_nwf_probs = acl_nwf_counts.divide(np.sum(acl_nwf_counts['count'])).to_dict()['count']

    print(" -acl_ncl")
    acl_ncl_counts = pd.pivot_table(pairs[['count', 'acl_ncl']], index=['acl_ncl'], values=['count', 'acl_ncl'], aggfunc=np.sum)
    acl_ncl_probs = acl_ncl_counts.divide(np.sum(acl_ncl_counts['count'])).to_dict()['count']


    print("expanding pairs ...")
    pairs = pairs.reindex(pairs.index.repeat(pairs['count']))
    del pairs['count']

    print("mapping adj to nouns ...")
    print(" -awf_to_nwfs")
    awf_to_nwfs = {k: g["nwf"].tolist() for k,g in pairs.groupby('awf')}

    print(" -awf_to_ncls")
    awf_to_ncls = {k: g["ncl"].tolist() for k,g in pairs.groupby('awf')}    

    print(" -acl_to_nwfs")
    acl_to_nwfs = {k: g["nwf"].tolist() for k,g in pairs.groupby('acl')}

    print(" -acl_to_ncls")
    acl_to_ncls = {k: g["ncl"].tolist() for k,g in pairs.groupby('acl')}

    print("mapping nouns to adjs ...")
    print(" -nwf_to_awfs")
    nwf_to_awfs = {k: g["awf"].tolist() for k,g in pairs.groupby('nwf')}

    print(" -nwf_to_acls")
    nwf_to_acls = {k: g["acl"].tolist() for k,g in pairs.groupby('nwf')}        

    print(" -ncl_to_awfs")
    ncl_to_awfs = {k: g["awf"].tolist() for k,g in pairs.groupby('ncl')}

    print(" -ncl_to_acls")
    ncl_to_acls = {k: g["acl"].tolist() for k,g in pairs.groupby('ncl')}

    print("calculating entropies ...")
    print(" -awf_nwf")
    awf_nwf_ents = ic(awf_to_nwfs)
    
    print(" -awf_ncl")
    awf_ncl_ents = ic(awf_to_ncls)
    
    print(" -acl_nwf")
    acl_nwf_ents = ic(acl_to_nwfs)

    print(" -acl_ncl")
    acl_ncl_ents = ic(acl_to_ncls)

    print("printing output ...")
    outfile = open("results.csv", 'w')
    outfile.write("id,idx,count,awf,nwf,acl,ncl,p_awf,p_acl,p_nwf,p_ncl,p_awf_nwf,p_awf_ncl,p_acl_nwf,p_acl_ncl,ic_awf_nwf,ic_awf_ncl,ic_acl_nwf,ic_acl_ncl,pmi_awf_nwf,pmi_awf_ncl,pmi_acl_nwf,pmi_acl_ncl,vd_awf_nwf\n")
    n = len(triples)
    for i, triple in enumerate(triples):
        print_progress(i+1, n)
        nwf = triple[3]
        ncl = None
        try:
            ncl = str(ncls[nwf])
        except:
            pass
        for j, awf in enumerate(triple[1:3]):
            acl = None
            try:
                acl = str(acls[awf])
            except:
                pass

            # probabilities
            p_awf = None
            p_acl = None
            p_nwf = None
            p_ncl = None
            p_awf_nwf = None
            p_awf_ncl = None
            p_acl_nwf = None
            p_acl_ncl = None            
            try:
                p_awf = awf_probs[awf]
            except:
                pass
            try:
                p_acl = acl_probs[acl]
            except:
                pass
            try:
                p_nwf = nwf_probs[nwf]
            except:
                pass
            try:
                p_ncl = ncl_probs[ncl]
            except:
                pass            
            try:
                nwf2awf = nwf_to_awfs[nwf]
                p_awf_nwf = Counter(nwf2awf)[awf]/len(nwf2awf)
            except:
                pass
            try:
                nwf2acl = nwf_to_acls[nwf]
                p_awf_ncl = Counter(nwf2awf)[awf]/len(nwf2acl)
            except:
                pass
            try:
                nwf2acl = nwf_to_acls[nwf]
                p_acl_nwf = Counter(nwf2acl)[acl]/len(nwf2acl)
            except:
                pass
            try:
                ncl2acl = ncl_to_acls[ncl]
                p_acl_ncl = Counter(ncl2acl)[acl]/len(ncl2acl)
            except:
                pass

            # integration cost
            ic_awf_nwf = None
            ic_awf_ncl = None
            ic_acl_nwf = None
            ic_acl_ncl = None
            try:
                ic_awf_nwf = awf_nwf_ents[awf]
            except:
                pass
            try:
                ic_awf_ncl = awf_ncl_ents[awf]
            except:
                pass
            try:
                ic_acl_nwf = acl_nwf_ents[acl]
            except:
                pass
            try:
                ic_acl_ncl = acl_ncl_ents[acl]
            except:
                pass

            # pmi
            pmi_awf_nwf = None
            pmi_awf_ncl = None
            pmi_acl_nwf = None
            pmi_acl_ncl = None
            try:
                pmi_awf_nwf = calc_pmi(float(awf_probs[awf]), float(nwf_probs[nwf]), float(awf_nwf_probs[awf + "_" + nwf]))
            except:
                pass
            try:
                pmi_awf_ncl = calc_pmi(float(awf_probs[awf]), float(ncl_probs[ncl]), float(awf_ncl_probs[awf + "_" + ncl]))
            except:
                pass
            try:
                pmi_acl_nwf = calc_pmi(float(acl_probs[acl]), float(nwf_probs[nwf]), float(acl_nwf_probs[acl + "_" + nwf]))
            except:
                pass

            try:
                pmi_acl_ncl = calc_pmi(float(acl_probs[acl]), float(ncl_probs[ncl]), float(acl_ncl_probs[acl + "_" + ncl]))
            except:
                pass

            # vector cosine distance
            vd_awf_nwf = None
            try:
                vd_awf_nwf = cosine(awf_vectors[awfs.index(awf)], nwf_vectors[nwfs.index(nwf)])
            except:
                pass


            outfile.write(str(i) + "," + str(j) + "," + str(triple[0]) + "," + awf + "," + nwf + "," + str(acl) + "," + str(ncl) + "," + str(p_awf) + "," + str(p_acl) + "," + str(p_nwf) + "," + str(p_ncl) + "," + str(p_awf_nwf) + "," + str(p_awf_ncl) + "," + str(p_acl_nwf) + "," + str(p_acl_ncl) + "," + str(ic_awf_nwf) + "," + str(ic_awf_ncl) + "," + str(ic_acl_nwf) + "," + str(ic_acl_ncl) + "," + str(pmi_awf_nwf) + "," + str(pmi_awf_ncl) + "," + str(pmi_acl_nwf) + "," + str(pmi_acl_ncl) + "," + str(vd_awf_nwf) + "\n")
                
    outfile.close()
    
print("")