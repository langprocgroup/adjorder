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
    parser.add_argument('inputfile', type=str, help='tab-separated list of X and y')
    args = parser.parse_args()

    combo = True
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    p_index = 0
    pmi_index = 0
    ic_index = 0
    vd_index = 0

    df = load_file(args.inputfile)
    clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs')
    plt.figure(1,figsize=(20,12))  
    if combo:
        plt.rcParams["figure.figsize"][0] = 12
        
    for i, predictor in enumerate(df.predictor.unique()):
        print(predictor)        
        X = []
        y = []

        X = np.array(df.loc[df['predictor'] == predictor]['diff'].values).astype('float').reshape(-1,1)
        y = np.array(df.loc[df['predictor'] == predictor]['result'].values).astype('int')

        if np.average(y) < 0.5:
            y = 1 - y

        xmax = np.max(X)
        
        if combo:
            X = X/xmax
            xmax = 1
        else:
            #plt.figure(1,figsize=(3,3))              
            plt.subplot(3, 5, (i+1))
            plt.axhline(0.5, color='gray', linestyle='--')
            #plt.clf()
            
        plt.xlim(0,xmax)
        plt.ylim(-0.1, 1.1)
        
        clf.fit(X,y)

        if not combo:
            plt.scatter(X,y,color='gray', marker='|')
        X_test = np.linspace(0, xmax, 300)

        y_test = model(X_test * clf.coef_ + clf.intercept_).ravel()
        auc = metrics.auc(X_test, y_test)/xmax #normalize curves to [0,1]

        title = predictor.split("_")
        if title[0] == 'p':
            try:
                title = "p(" + title[1] + "|" + title[2] + ")"
            except:
                title = "p(" + title[1] + ")"
            linestyle = 'solid'
            color = colors[p_index]
            p_index += 1
        elif title[0] =='pmi':
            title = "pmi(" + title[1] + ";" + title[2] + ")"
            linestyle='dotted'
            color = colors[pmi_index]
            pmi_index += 1
        elif title[0] == "ic":
            title = "ic(" + title[1] + "," + title[2] + ")"
            linestyle='dashed'
            color = colors[ic_index]
            ic_index += 1
        elif title[0] == "vd":
            title = "vd(" + title[1] + ";" + title[2] + ")"
            linestyle='dashdot'
            color = colors[vd_index]
            vd_index += 1

        if not combo:
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
        else:
            plt.plot(X_test, y_test, color=color, linestyle=linestyle, label="{:.4f}".format(auc) + " " + title)



            
        

        if not combo:
            plt.title(title + ", n: " + str(n))
            plt.xlabel("delta")
            plt.ylabel("probability")
            plt.legend(loc = 'center left', bbox_to_anchor=(0, 0.2))
    if combo:
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title="AUC")
        plt.title("all predictors, n: " + str(len(X)))
        plt.xlabel("delta")
        plt.ylabel("probability")
        plt.savefig("all_predictors.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.tight_layout()
        plt.savefig("individual_predictors.png", bbox_inches='tight', pad_inches=0)
