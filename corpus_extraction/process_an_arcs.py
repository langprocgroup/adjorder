#!/usr/bin/python3
""" Take a file of (A-)A-N arcs and filter them and sum up the counts"""
from __future__ import print_function
import sys
import gzip
from collections import namedtuple, Counter

VERBOSE = 1

Word = namedtuple('Word', "word pos".split())

ADJ_PATH = "../data/english_adjectives.txt"
NOUN_PATH = "../data/english_nouns.txt"

with open(ADJ_PATH) as infile:
    ADJECTIVES = {line.strip() for line in infile}
with open(NOUN_PATH) as infile:
    NOUNS = {line.strip() for line in infile}

def rep(word):
    return "/".join(word)

def is_noun(word, celex=True):
    pos_ok = word.pos.startswith('NN')
    celex_ok = not celex or (word.word.lower() in NOUNS or word.word + "s" in NOUNS)
    if VERBOSE > 1:
        if pos_ok and not celex_ok:
            print("Non-celex noun:", word.word, file=sys.stderr)
    return pos_ok and celex_ok

def is_adjective(word, celex=True):
    pos_ok = word.pos.startswith('JJ')
    celex_ok = not celex or word.word.lower() in ADJECTIVES
    if VERBOSE > 1:
        if pos_ok and not celex_ok:
            print("Non-celex adjective:", word.word, file=sys.stderr)
    return pos_ok and celex_ok

def parse_word(word):
    word, pos, _, _ = word.rsplit("/", 3)
    return Word(word, pos)

def process_an(parts):
    word1, word2, count = parts
    count = int(count)
    adj = parse_word(word1)
    head = parse_word(word2)
    if is_noun(head) and is_adjective(adj):
        return (head, adj), count
    else:
        return None

def process_aan(parts):
    word1, word2, word3, count = parts
    count = int(count)
    adj1 = parse_word(word1)
    adj2 = parse_word(word2)
    head = parse_word(word3)
    if is_noun(head) and is_adjective(adj1) and is_adjective(adj2):
        return (head, adj1, adj2), count
    else:
        return None
    
def read_lines(lines):
    for i, line in enumerate(lines):
        if VERBOSE and i % 10000000 == 0:
            print(i, file=sys.stderr)
        parts = line.strip().split()
        if len(parts) == 3:
            result = process_an(parts)
        elif len(parts) == 4:
            result = process_aan(parts)
        else:
            print("Bad line: %s" % line.strip(), file=sys.stderr)
        if result:
            yield result

def sum_counts(arcs_and_counts):
    c = Counter()
    for arc, count in arcs_and_counts:
        c[arc] += count
    return c

def main(filename=None):
    if filename:
        infile = gzip.open(filename, mode='rt')
    else:
        infile = sys.stdin
    counts = read_lines(infile)
    c = sum_counts(counts)
    for (head, *adjs), count in c.items():
        adjs = "\t".join(map(rep, adjs))
        print("\t".join([rep(head), adjs, str(count)]))

if __name__ == '__main__':
    main(*sys.argv[1:])        
    
