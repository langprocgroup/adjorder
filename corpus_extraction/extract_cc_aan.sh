#!/usr/bin/env bash
zcat ~/data/openmind/parsing/*.gz | python extract_cc_aan.py $1 | sort | uniq -c | awk -F' ' '{print $2 "\t" $3 "\t" $4 "\t" $1}'
