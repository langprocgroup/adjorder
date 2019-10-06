"""
1. reads in a tab-delimited list of triples
  NOUN ADJ ADJ COUNT
2. calculates pmi of each adj-noun
3. counts how many of the triples arranged such that highest-pmi adj is closest
"""

import sys, codecs, numpy, pickle, os, re, math
import pandas as pd
#from scipy.stats import entropy
#from collections import Counter
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

def pmi(pa, pb, pab):
    return math.log(pab/float(pa*pb),2) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='integration cost')
    parser.add_argument('triples', type=str, help='file containing [count adj adj noun] triples')    
    args = parser.parse_args()

    pairs = None
    triples = None
    triples_pickle_file = "triples.pickle"

    if triples_pickle_file in os.listdir():
        print("loading triples data from " + triples_pickle_file + " ...")
        [pairs, triples] = load_from_pickle(triples_pickle_file, [pairs, triples])
    else:
        print("loading triples data from " + args.triples + " ...")
        pairs, triples = load_data(args.triples)
        print("\nsaving triples data to " + triples_pickle_file + " ...")
        save_to_pickle(triples_pickle_file, [pairs, triples])
    adjs = [row[0] for row in pairs] #don't remove duplicates
    nouns = [row[1] for row in pairs]    

    adj_probs = pd.DataFrame(adjs, columns=['adj'])['adj'].value_counts().divide(len(adjs))
    noun_probs = pd.DataFrame(nouns, columns=['noun'])['noun'].value_counts().divide(len(nouns))
    pair_probs = pd.DataFrame(pairs, columns=['adj', 'noun'])
    pair_probs['pair'] = pair_probs['adj'] + ":" + pair_probs['noun']
    del pair_probs['adj']
    del pair_probs['noun']        
    pair_probs = pair_probs['pair'].value_counts().divide(len(pairs))

    print("calculating pmi ...")
    correct = 0
    incorrect = 0
    tie = 0
    
    for i, triple in enumerate(triples):
        print_progress(i+1, len(triples))
    
        adj1 = pmi(float(adj_probs[triple[0]]), float(noun_probs[triple[2]]), float(pair_probs[triple[0] + ":" + triple[2]]))
        adj2 = pmi(float(adj_probs[triple[1]]), float(noun_probs[triple[2]]), float(pair_probs[triple[1] + ":" + triple[2]]))


        if adj1 < adj2:
            correct += 1
        elif adj1 > adj2:
            incorrect += 1
        else:
            tie += 1


    #print("\n\n*** counts ***")
    #print("   adjs: " + str(len(adjs)))
    #print("  nouns: " + str(len(nouns)))
    #print("triples: " + str(len(triples)))

    print("\n\n*** results ***")
    total = incorrect + correct
    print(str(correct/total) + " (" + str(correct) + "/" + str(total) + ") ties: " + str(tie))
    
