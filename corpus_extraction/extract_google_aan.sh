CUTOFF=1960
zcat ~/data/openmind/syntngrams/biarcs.*-of-99.gz | grep -e "/amod/3.*/amod/3.*/NN.*/0" | python filter_corpus.py 3 $CUTOFF | python process_an_arcs.py > aan_filtered.tsv
