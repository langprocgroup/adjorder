#!/usr/bin/env bash
sort | uniq -c | awk -F' ' '{print $2 "\t" $3 "\t" $4 "\t" $1}'
