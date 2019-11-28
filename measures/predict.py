import subprocess, argparse, sys
import numpy as np
import pandas as pd

def load_scores(filename):
    df = pd.read_csv(filename, sep=",")
    return df

def print_progress(i, n):
    j = (i+1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*int(20*j), 100*j, i, n))
    sys.stdout.flush()
    return i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate deltas for each predictor')
    parser.add_argument('scores', type=str, help='comma-separated scores file')
    args = parser.parse_args()
    
    scores = load_scores(args.scores)

    n_meta = 7
    zeros = np.zeros((1, scores.shape[1] - n_meta)).astype(int)
    
    correct = pd.DataFrame(data=zeros, columns=scores.columns[n_meta:])
    incorrect = correct.copy()
    unknown = correct.copy()
    tie = correct.copy()

    n = scores.shape[0]
    outfile = open("deltas.csv", 'w')
    outfile.write("id,predictor,delta,result\n")
    
    for i in range(0, scores.shape[0], 2): # for each AAN triple...
        row1 = scores.iloc[i] # the attested first adjective 
        row2 = scores.iloc[i+1] # the attested second adjective

        if row1['id'] != row2['id'] or row1['idx'] != 0 or row2['idx'] != 1:
            print("something's wrong with order of triples")
            print(row1)
            print(row2)
            exit()

        # correct, unknown, tie, and incorrect are dataframes
        # where each column is a predictor, each row is an AAN triple, and the value is the count
        count = int(row1['count']) 
        for col in correct.columns: 
            if row1[col] == 'None' or row2[col] == 'None':
                unknown[col] += count
            elif row1[col] == row2[col]:
                tie[col] += count
            else:
                if float(row1[col]) > float(row2[col]): # if S(a1, n) > S(a2, n), say predicted=1
                    predicted = '1'
                    correct[col] += count
                else: # otherwise say predicted=0 and count it as wrong
                    predicted = '0'
                    incorrect[col] += count
                for _ in range(count): # write the number of rows corresponding to the count
                    outfile.write(str(row1['id']) + "," + col + "," + str(float(row1[col]) - float(row2[col])) + "," + str(predicted) + "\n")
            print_progress(i+1, n)
    outfile.close()

    print("")
            
    for col in correct.columns:
        p = correct[col].values[0]
        f = incorrect[col].values[0]
        u = unknown[col].values[0]
        t = tie[col].values[0]
        pattern = "(high low noun)"

        total = p + f + t
        
        if f > p:
            a = p
            p = f
            f = a
            pattern = "(low high noun)"
        if p > 0:
            print("\n" + col + "\n" + pattern)
            print("pass: {:.4f}".format(p/(total)) + " (" + str(p) + "/" + str(total) + ")")
            print("fail: {:.4f}".format(f/(total)) + " (" + str(f) + "/" + str(total) + ")")            
            print(" tie: {:.4f}".format(t/(total)) + " (" + str(t) + "/" + str(total) + ")")
