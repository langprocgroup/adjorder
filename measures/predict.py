import subprocess, argparse, sys
import numpy as np
import pandas as pd

def load_results(filename):
    df = pd.read_csv(filename, sep=",")
    return df

def print_progress(i, n):
    j = (i+1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*int(20*j), 100*j, i, n))
    sys.stdout.flush()
    return i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict triples')
    parser.add_argument('results', type=str, help='comma-separated results file')
    args = parser.parse_args()
    
    results = load_results(args.results)

    n_meta = 7
    zeros = np.zeros((1, results.shape[1] - n_meta)).astype(int)
    
    correct = pd.DataFrame(data=zeros, columns=results.columns[n_meta:])
    incorrect = correct.copy()
    unknown = correct.copy()
    tie = correct.copy()

    n = results.shape[0]
    outfile = open("output.csv", 'w')
    outfile.write("id,predictor,diff,result\n")
    
    for i in range(0, results.shape[0], 2):
        row1 = results.iloc[i]
        row2 = results.iloc[i+1]

        if row1['id'] != row2['id'] or row1['idx'] != 0 or row2['idx'] != 1:
            print("something's wrong with order of triples")
            print(row1)
            print(row2)
            exit()

        count = int(row1['count'])
        for col in correct.columns:
            if row1[col] == 'None' or row2[col] == 'None':
                unknown[col] += count
            elif row1[col] == row2[col]:
                tie[col] += count
            else:
                predicted = False
                if row1[col] > row2[col]:
                    predicted = True

                if predicted:
                    correct[col] += count
                else:
                    incorrect[col] += count
                outfile.write(str(row1['id']) + "," + col + "," + str(abs(float(row1[col]) - float(row2[col]))) + "," + str(predicted).replace("True", "1").replace("False", "0") + "\n")
            try:
                pct = int(correct[col].values[0])/(int(correct[col].values[0]) + int(incorrect[col].values[0]))
            except:
                pass
            print_progress(i+1, n)

    outfile.close()

    print("")
            
    for col in correct.columns:
        p = correct[col].values[0]
        f = incorrect[col].values[0]
        u = unknown[col].values[0]
        t = tie[col].values[0]

        if f > p:
            a = p
            p = f
            f = a
        if p > 0:
            print("\n" + col)
            print("correct: {:.4f}".format(p/(p+f)) + " (" + str(p) + "/" + str(p+f) + ")")
            print("    tie: {:.4f}".format(t/(p+f+u+t)) + " (" + str(t) + "/" + str(p+f+u+t) + ")")
            print("unknown: {:.4f}".format(u/(p+f+u+t)) + " (" + str(u) + "/" + str(p+f+u+t) + ")")



        

    
