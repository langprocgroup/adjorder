Calculate predictors of adjective order and test them in large dependency treebanks.

The directory `corpus_extraction` has scripts for pulling relevant data out of CoNLLU-formatted dependency treebank files. Supposing you have a bunch of files at location `$CORPORA`, run the following to get all the adjective--adjective--noun pairs:

```{bash}
cat $CORPORA | python extract_cc.py aan > output_filename
```

and run the following to just get all the adjectives:

```{bash}
cat $CORPORA | python extract_cc.py a > output_filename
```

Data in `data/english_adjectives.txt` and `data/english_nouns.txt` is from CELEX.

If you are here for the code used in Futrell (2019), check out the previous version of this repo at 464e24d.
