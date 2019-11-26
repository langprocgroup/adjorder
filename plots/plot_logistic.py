import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
import subprocess, argparse, sys, random

def load_file(filename):
    df = pd.read_csv(filename, sep=",")
    return df

def model(x):
    return 1/(1 + np.exp(-x))

def print_progress(i, n):
    j = (i+1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*int(20*j), 100*j, i, n))
    sys.stdout.flush()
    return i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot logistic')
    parser.add_argument('inputfile', type=str, help='comma-separated file with [id,predictor,delta,result]')
    args = parser.parse_args()

    df = load_file(args.inputfile)
    clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs')
    plt.figure(1,figsize=(20,12))  
        
    for i, predictor in enumerate(df.predictor.unique()):
        X = []
        y = []

        X = np.array(df.loc[df['predictor'] == predictor]['delta'].values).astype('float').reshape(-1,1)
        y = np.array(df.loc[df['predictor'] == predictor]['result'].values).astype('int')

        if np.average(y) < 0.5:
            y = 1 - y

        xmax = np.max(X)
        
        plt.subplot(4, 5, (i+1))
        plt.axhline(0.5, color='gray', linestyle='--')
            
        plt.xlim(0,xmax)
        plt.ylim(-0.1, 1.1)
        
        clf.fit(X,y)

        plt.scatter(X,y,color='gray', marker='|')
        X_test = np.linspace(0, xmax, 300)
        y_test = model(X_test * clf.coef_ + clf.intercept_).ravel()
        auc = metrics.auc(X_test, y_test)/xmax #normalize auc to [0,1] for comparison

        title = predictor.split("_")
        if title[0] == 'p':
            try:
                title = "p(" + title[1] + "|" + title[2] + ")"
            except:
                title = "p(" + title[1] + ")"
        elif title[0] =='pmi':
            title = "pmi(" + title[1] + ";" + title[2] + ")"
        elif title[0] == "ic":
            title = "ic(" + title[1] + "," + title[2] + ")"
        elif title[0] == "ig":
            title = "ig(" + title[1] + "," + title[2] + ")"
        elif title[0] == "s":
            title="subj(" + title[1] + ")"

        color='green'
        linestyle='solid'
        p = np.sum(y)
        n = len(y)
        f = n - p
        if p > f:
            s = p
        else:
            s = f

        plt.axhline(s/n, color='blue', linestyle='--', label="acc: {:.4f}".format(s/n))
        plt.plot(X_test, y_test, color=color, linestyle=linestyle, label="auc: {:.4f}".format(auc))

        plt.title(title + ", n: " + str(n))
        plt.xlabel("delta")
        plt.ylabel("probability")
        plt.legend(loc = 'center left', bbox_to_anchor=(0, 0.25))
        print(title + "\tn: cp " + str(n) + "\tacc: {:4f}\tauc: {:4f}".format(s/n, auc)) 
        plt.tight_layout()
        plt.savefig("predictors.png", bbox_inches='tight', pad_inches=0)
