#!/usr/bin/env bash
head -n 1 $1 | awk '{print "count," $1}'
cat $1 | sed "1d" | sort | uniq -c | awk -F' ' '{print $1 "," $2}'
