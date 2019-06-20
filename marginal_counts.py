import sys
from collections import Counter

def main(filename, col):
    col = int(col)
    c = Counter()
    with open(filename) as infile:
        for line in infile:
            parts = line.strip().split("\t")
            c[parts[col]] += int(parts[-1])
    for key, value in c.items():
        print("\t".join([str(key), str(value)]))

if __name__ == '__main__':
    main(*sys.argv[1:])
            
            
