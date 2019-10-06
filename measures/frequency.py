"""
1. reads in a tab-delimited list of triples
  NOUN ADJ ADJ COUNT
2. counts how many of the triples arranged such that most frequent adj is first
"""

import sys, codecs, numpy, pickle, os, re
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

    df = pd.DataFrame(adjs, columns=['adj'])['adj'].value_counts()
    
    print("calculating frequencies (counts) ...")
    correct = 0
    incorrect = 0
    tie = 0
    for i, triple in enumerate(triples):
        print_progress(i, len(triples))
        try:
            adj1 = df[triple[0]]
            adj2 = df[triple[1]]

            if adj1 > adj2:
                correct += 1
                #if correct == 1:
                #    print("correct: " + str(triple) + " (" + str(adj1) + " > " + str(adj2) + ")")
            elif adj1 < adj2:
                incorrect += 1
                #if incorrect == 1:
                #    print("incorrect: " + str(triple) + " (" + str(adj1) + " < " + str(adj2) + ")")                    
            else:
                tie += 1
        except:
            pass
        

    #print("\n\n*** counts ***")
    #print("   adjs: " + str(len(adjs)))
    #print("  nouns: " + str(len(nouns)))
    #print("triples: " + str(len(triples)))

    print("\n\n*** results ***")
    total = incorrect + correct
    print(str(correct/total) + " (" + str(correct) + "/" + str(total) + ") ties: " + str(tie))
    
