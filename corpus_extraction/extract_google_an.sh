CUTOFF=1960
zcat ~/data/openmind/syntngrams/arcs.*-of-99.gz | grep -e "/amod/2.*/NN.*/0" | python filter_corpus.py 2 $CUTOFF | python process_an_arcs.py
