import pandas as pd
import numpy as np
import sys

def load_ud_file(filename):
    ud = pd.read_csv(filename, sep="\t",error_bad_lines=False, engine='python', header=None, comment= '#', quoting=3)
    return ud

def find_aan(df):
    aan = []
    chunksize = 3
    chunk = [x[:] for x in [[None] * df.shape[1]] * chunksize]
    for index, row in df.iterrows():
        chunk[2] = row
        try:
            if chunk[0][3] == "ADJ" and chunk[1][3] == "ADJ" and chunk[2][3] == "NOUN" and int(chunk[1][0]) == int(chunk[0][0]) + 1 and chunk[1][3] == chunk[0][3] and chunk[1][6] == chunk[0][6] and int(chunk[2][0]) == int(chunk[1][0]) + 1:
                aan.append([str(chunk[0][1]).lower(), str(chunk[1][1]).lower(), str(chunk[2][1]).lower()])
        except:
            pass

        chunk[0] = chunk[1]
        chunk[1] = chunk[2]

    return aan

def count_aan(aan):
    df = pd.DataFrame(aan, columns=['adj1', 'adj2', 'noun'])
    return df.groupby(['adj1', 'adj2', 'noun']).size().reset_index(name='count')

print("load UD data from " + sys.argv[1] + "...")
ud = load_ud_file(sys.argv[1])

print("find aan...")
aan = find_aan(ud)

print("count aan...")
c_aan = count_aan(aan)

outfile = open("triples.tsv", "a")


for index, row in c_aan.iterrows():
    outfile.write(str(row[3]) + "\t" + str(row[0]) + "\t" + str(row[1]) + "\t" + str(row[2]) + "\n")
outfile.close()
