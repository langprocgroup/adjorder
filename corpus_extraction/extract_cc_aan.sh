#!/usr/bin/env bash
zcat ~/data/openmind/parsing/*.gz | python extract_cc.py aan $1 
