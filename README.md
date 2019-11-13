Calculate predictors of adjective order and test them in large dependency treebanks.

## Data

### External data requirements

In this study, we extracted data from the following external data sources, not included here:

-- A parsed Common Crawl corpus in CoNLLU format, introduced in Futrell et al. (2019).
-- Universal Dependencies 2.4, in particular the English Web Treebank
-- GLoVe embeddings

### Data provided

English adjective and noun wordforms from CELEX are provided in `data/english_adjectives.txt` and `data/english_nouns.txt` is from CELEX.

The files `data/subjectivity*` are from Scontras et al. (2017): these are subjectivity ratings collected in previous experiments.

The subjectivity ratings collected for this study are at `experiments/1-UD-subjectivity/results/adjective-subjectivity.csv`.

## Extracting adjective data from corpora

The directory `corpus_extraction` has scripts for pulling relevant data out of CoNLLU-formatted dependency treebank files. Supposing you have a bunch of files at location `$CORPORA`, run the following in the directory `corpus_extraction` to get all the adjective--adjective--noun pairs:

```{bash}
cat $CORPORA | python extract_conllu.py aan > aan.csv
sh csvcount.sh aan.csv > aan_counts.csv
```

and run the following to just get all the adjectives:

```{bash}
cat $CORPORA | python extract_conllu.py a > a.csv
sh csvcount.sh a.csv > a_counts.csv
```

## Calculating predictors

Predictors are calculated using `measures/score_adj_pairs.py`. 

## Previous work

If you are here for the code used in Futrell (2019), check out the previous version of this repo at #464e24d.
