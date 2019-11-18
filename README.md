Calculate predictors of adjective order and test them in large dependency treebanks.

## Data

### External data requirements

In this study, we extracted data from the following external data sources, not included here:

* A parsed Common Crawl corpus in CoNLLU format, introduced in Futrell et al. (2019).
* Universal Dependencies 2.4, in particular the English Web Treebank
* GLoVe embeddings from [glove.42B.300d.zip](http://nlp.stanford.edu/data/glove.42B.300d.zip)

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
## Choosing a k for clusters


## Calculating predictors

Predictors are calculated using `measures/score_adj_pairs.py` with the following arguments:
* `-v $GLOVE` -- a file containing space-delimited wordforms and their vectors
* `-p $PAIRS` -- a file containing tab-delimited `count adj noun` pairs
* `-t $TRIPLES` -- a file containing tab-delimited `count adj adj noun` triples
* `-s $SUBJ` -- a file containing tab-delimied `adj subj_rating` pairs, generated by `cut -d',' -f2,4 experiments/1-UD-subjectivity/results/adjective-subjectivity.csv | tr -d '"' | tr ',' '\t'`

Output is a comma-delimited `scores.csv` with the following columns:
1. `id` -- the ID of a triple in `$TRIPLES`
1. `idx` -- 0 or 1 depending on position of this adjective in `$TRIPLES`
1. `count` -- the count of this triple inn `$TRIPLES`
1. `awf` -- adjective wordform
1. `nwf` -- noun wordform
1. `acl` -- adjective cluster ID
1. `ncl` -- noun cluster ID
1. various predictors named according to the following scheme:
    * `p_` -- log probability
    * `ic_` -- integration cost
    * `pmi_` -- pointwise mutual information
    * `vd_` -- vector distance
    * `subj_` -- subjectivity rating

Note that `score_adj_pairs.py` creates a pickle file `data.pkl` to store adjective and noun vectors and clusters, since generating these is the most time-consuming process. The pickled data will be used on subsequent runs of the script.

## Running predictors

The predictors calculated and reported in `scores.csv` can be run with `python measures/predict.py scores.csv`. Output is `output.csv`, a comma-delimited file with the following columns:
1. `id` -- the ID of a triple in `$TRIPLES`
1. `predictor` -- the predictor being run
1. `diff` -- the (absolute) difference between the predictor score for each adjective
1. `result` -- whether the adjective with the smallest predictor comes first (0) or second (1).

Note that predictors with `None` values in `scores.csv` will not be included in `output.csv`.

## Generating plots

Plots can be generated by running `plots/plot_logistic.py output.csv`. A single image (`individual_predictors.png`) will be generated with a plot for each predictor, showing predictive accuracy and area under curve (AUC) for a logistic regression indicating the predicted probability (y-axis) as a function of the difference between each adjective's score (x-axis). Note that if accuracy is less than 0.5 for a given predictor, the polarity of the predictions -- and the resulting logistic regression -- is switched.

## Previous work

If you are here for the code used in Futrell (2019), check out the previous version of this repo at #464e24d.
