import sys, codecs, os, math, argparse
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter

def print_progress(i, n):
    j = (i+1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% N %d/%d " % ('='*int(20*j), 100*j, i, n))
    sys.stdout.flush()
    return i + 1

def load_pairs(filename):
    pairs = pd.read_csv(filename, sep=",", dtype=str)
    pairs['awf'] = pairs['awf'].str.lower()
    pairs['nwf'] = pairs['nwf'].str.lower()
    pairs['count'] = pd.to_numeric(pairs['count'])
    return pairs

def load_test_data(filename, pair_acls, pair_ncls):
    test_data = {}
    awfs = []
    nwfs = []
    acls = []
    ncls = []
    df = pd.read_csv(filename, sep=",", dtype=str)[['count', 'adj1_word', 'adj2_word', 'noun_word']]
    for w in ['noun_word', 'adj1_word', 'adj2_word']:
        df[w] = df[w].str.lower()
    df['count'] = pd.to_numeric(df['count'])
    df = df.fillna('null') # treat instances of 'null' in df as strings, not NULL
    test_data['triples'] = df.values.tolist()
    test_data['awfs'] = np.unique(df[['adj1_word', 'adj2_word']].values)
    test_data['nwfs'] = np.unique(df['noun_word'].values)
    test_data['acls'] = pd.merge(pd.DataFrame(test_data['awfs']), pd.DataFrame.from_dict(pair_acls, orient='index'), how='inner', left_on=[0], right_index=True)['0_y'].unique()
    test_data['ncls'] = pd.merge(pd.DataFrame(test_data['nwfs']), pd.DataFrame.from_dict(pair_ncls, orient='index'), how='inner', left_on=[0], right_index=True)['0_y'].unique()

    return test_data

def load_vectors(filename):
    df = pd.read_csv(filename, sep='["]* ["]*', header=None, error_bad_lines=False, engine='python')
    return df

def load_subj(filename):
    df = pd.read_csv(filename, sep=",")
    return df

def calc_pmi(pxy, px):
    #return math.log(pxy/px,2)
    return pxy-px

def info_theory(d, n, start_ent):
    integ_cost = {}
    info_gain = {}
    for k in d:
        dist = list(Counter(d[k]).values())
        ent = entropy(dist, base=2)
        integ_cost[k] = ent
        info_gain[k] = start_ent - (len(dist)/n) * ent
    return integ_cost, info_gain
        
if __name__ == '__main__':
    # awf = adjective wordform
    # acl = adjective cluster
    # nwf = noun wordform
    # ncl = noun cluster
    
    parser = argparse.ArgumentParser(description='score adj pairs')
    parser.add_argument('-p', '--pairs', nargs=1, dest='pairs', required=True, help='comma-delimited file containing [count,awf,nwf,acl,ncl] for calculating predictors')
    parser.add_argument('-t', '--test', nargs=1, dest='test_file', required=True, help='comma-delimited file containing [count,adj1_word,adj2_word,noun_word] for test')
    parser.add_argument('-s', '--subj', nargs=1, dest='subj', required=True, help='comma-delimieted file containing [adj,subj] subjectivity ratings')
    args = parser.parse_args()
    
    print("loading pairs data from " + args.pairs[0] + " ...")
    pairs = load_pairs(args.pairs[0])

    print("making cluster dicts ...")
    adf = pairs[['awf', 'acl']]
    acls = adf.set_index(['awf']).to_dict()['acl']
    ndf = pairs[['nwf', 'ncl']]
    ncls = ndf.set_index(['nwf']).to_dict()['ncl']

    print("loading test data from " + args.test_file[0] + " ...")
    test_data = load_test_data(args.test_file[0], acls, ncls)

    print("loading subjectivities ...")
    subjectivities = load_subj(args.subj[0])[['predicate', 'response']].set_index('predicate')
    subj = subjectivities.to_dict()['response']

    print("adding pairs to dataframe ...")
    pairs['awf_nwf'] = pairs['awf'] + "_" + pairs['nwf']
    pairs['awf_ncl'] = pairs['awf'] + "_" + pairs['ncl']
    pairs['acl_nwf'] = pairs['acl'] + "_" + pairs['nwf']
    pairs['acl_ncl'] = pairs['acl'] + "_" + pairs['ncl']

    print("calculating cluster subjectivities ...")
    subj_clusters = pd.merge(subjectivities, pd.DataFrame.from_dict(acls, orient='index'), how='inner', left_index=True, right_index=True).reset_index()
    subj_clusters.columns = ['predicate', 'response', 'cluster']
    subj_clusters = pd.pivot_table(subj_clusters[['response', 'cluster']], index=['cluster'], values=['response', 'cluster'], aggfunc=np.average).to_dict()['response']

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
    awf_to_nwfs = {k: g["nwf"].tolist() for k,g in pairs.loc[pairs['awf'].isin(test_data['awfs'])].groupby('awf')}

    print(" -awf_to_ncls")
    awf_to_ncls = {k: g["ncl"].tolist() for k,g in pairs.loc[pairs['awf'].isin(test_data['awfs'])].groupby('awf')}

    print(" -acl_to_nwfs")
    acl_to_nwfs = {k: g["nwf"].tolist() for k,g in pairs.loc[pairs['acl'].isin(test_data['acls'])].groupby('acl')}

    print(" -acl_to_ncls")
    acl_to_ncls = {k: g["ncl"].tolist() for k,g in pairs.loc[pairs['acl'].isin(test_data['acls'])].groupby('acl')}

    print("mapping nouns to adjs ...")
    print(" -nwf_to_awfs")
    nwf_to_awfs = {k: g["awf"].tolist() for k,g in pairs.loc[pairs['nwf'].isin(test_data['nwfs'])].groupby('nwf')}

    print(" -nwf_to_acls")
    nwf_to_acls = {k: g["acl"].tolist() for k,g in pairs.loc[pairs['nwf'].isin(test_data['nwfs'])].groupby('nwf')}

    print(" -ncl_to_awfs")
    ncl_to_awfs = {k: g["awf"].tolist() for k,g in pairs.loc[pairs['ncl'].isin(test_data['ncls'])].groupby('ncl')}

    print(" -ncl_to_acls")
    ncl_to_acls = {k: g["acl"].tolist() for k,g in pairs.loc[pairs['ncl'].isin(test_data['ncls'])].groupby('ncl')}

    print("calculating entropies ...")
    print(" -awf_nwf")
    start_ent_nwf = entropy(list(Counter(pairs['nwf'].values).values()), base=2)
    n_nwf = len(test_data['nwfs'])
    awf_nwf_ic, awf_nwf_ig = info_theory(awf_to_nwfs, n_nwf, start_ent_nwf)
    
    print(" -awf_ncl")
    start_ent_ncl = entropy(list(Counter(pairs['ncl'].values).values()), base=2)
    n_ncl = len(test_data['ncls'])
    awf_ncl_ic, awf_ncl_ig = info_theory(awf_to_ncls, n_ncl, start_ent_ncl)
    
    print(" -acl_nwf")
    acl_nwf_ic, acl_nwf_ig = info_theory(acl_to_nwfs, n_nwf, start_ent_nwf)

    print(" -acl_ncl")
    acl_ncl_ic, acl_ncl_ig = info_theory(acl_to_ncls, n_ncl, start_ent_ncl)

    print("printing output to scores.csv ...")
    outfile = open("scores.csv", 'w')
    outfile.write("id,idx,count,awf,nwf,acl,ncl,p_awf,p_acl,p_nwf,p_ncl,p_awf_nwf,p_awf_ncl,p_acl_nwf,p_acl_ncl,ic_awf_nwf,ic_awf_ncl,ic_acl_nwf,ic_acl_ncl,pmi_awf_nwf,pmi_awf_ncl,pmi_acl_nwf,pmi_acl_ncl,s_awf,s_acl,ig_awf_nwf,ig_awf_ncl,ig_acl_nwf,ig_acl_ncl\n")
    n = len(test_data['triples'])
    for i, triple in enumerate(test_data['triples']):
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
                p_awf = math.log(awf_probs[awf],2)
            except:
                pass
            try:
                p_acl = math.log(acl_probs[acl],2)
            except:
                pass
            try:
                p_nwf = math.log(nwf_probs[nwf],2)
            except:
                pass
            try:
                p_ncl = math.log(ncl_probs[ncl],2)
            except:
                pass            
            try:
                nwf2awf = nwf_to_awfs[nwf]
                if p_awf != None: p_awf_nwf = math.log(Counter(nwf2awf)[awf]/len(nwf2awf),2)
            except:
                pass
            try:
                nwf2acl = nwf_to_acls[nwf]
                if p_awf != None: p_awf_ncl = math.log(Counter(nwf2acl)[acl]/len(nwf2acl),2)
            except:
                pass
            try:
                nwf2acl = nwf_to_acls[nwf]
                if acl != None: p_acl_nwf = math.log(Counter(nwf2acl)[acl]/len(nwf2acl),2)
            except:
                pass
            try:
                ncl2acl = ncl_to_acls[ncl]
                if acl != None: p_acl_ncl = math.log(Counter(ncl2acl)[acl]/len(ncl2acl),2)
            except:
                pass

            # integration cost
            ic_awf_nwf = None
            ic_awf_ncl = None
            ic_acl_nwf = None
            ic_acl_ncl = None
            try:
                ic_awf_nwf = awf_nwf_ic[awf]
            except:
                pass
            try:
                ic_awf_ncl = awf_ncl_ic[awf]
            except:
                pass
            try:
                ic_acl_nwf = acl_nwf_ic[acl]
            except:
                pass
            try:
                ic_acl_ncl = acl_ncl_ic[acl]
            except:
                pass

            # info gain
            ig_awf_nwf = None
            ig_awf_ncl = None
            ig_acl_nwf = None
            ig_acl_ncl = None
            try:
                ig_awf_nwf = awf_nwf_ig[awf]
            except:
                pass
            try:
                ig_awf_ncl = awf_ncl_ig[awf]
            except:
                pass
            try:
                ig_acl_nwf = acl_nwf_ig[acl]
            except:
                pass
            try:
                ig_acl_ncl = acl_ncl_ig[acl]
            except:
                pass

            # pmi
            pmi_awf_nwf = None
            pmi_awf_ncl = None
            pmi_acl_nwf = None
            pmi_acl_ncl = None
            try:
                pmi_awf_nwf = calc_pmi(p_awf_nwf, p_awf)
            except:
                pass
            try:
                pmi_awf_ncl = calc_pmi(p_awf_ncl, p_awf)
            except:
                pass
            try:
                pmi_acl_nwf = calc_pmi(p_acl_nwf, p_acl)
            except:
                pass

            try:
                pmi_acl_ncl = calc_pmi(p_acl_ncl, p_acl)
            except:
                pass
            
            # subjectivity
            s_awf = None
            s_acl = None
            try:
                s_awf = subj[awf]
            except:
                pass
            try:
                s_acl = subj_clusters[acl]
            except:
                pass

            outfile.write(str(i) + "," + str(j) + "," + str(triple[0]) + "," + awf + "," + nwf + "," + str(acl) + "," + str(ncl) + "," + str(p_awf) + "," + str(p_acl) + "," + str(p_nwf) + "," + str(p_ncl) + "," + str(p_awf_nwf) + "," + str(p_awf_ncl) + "," + str(p_acl_nwf) + "," + str(p_acl_ncl) + "," + str(ic_awf_nwf) + "," + str(ic_awf_ncl) + "," + str(ic_acl_nwf) + "," + str(ic_acl_ncl) + "," + str(pmi_awf_nwf) + "," + str(pmi_awf_ncl) + "," + str(pmi_acl_nwf) + "," + str(pmi_acl_ncl) + "," + str(s_awf) + "," + str(s_acl) + "," + str(ig_awf_nwf) + "," + str(ig_acl_nwf) + "," + str(ig_awf_ncl) + "," + str(ig_acl_ncl) + "\n")
                
    outfile.close()
    
print("")
