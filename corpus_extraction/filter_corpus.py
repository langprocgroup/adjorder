import sys

num_words = int(sys.argv[1])
cutoff = int(sys.argv[2])
lines = sys.stdin
sep = " "

for line in lines:
    word, words, number, *counts = line.split("\t")
    wordparts = words.split(" ")
    if len(wordparts) == num_words:
        counts = [part.split(",") for part in counts]
        count = sum(int(count) for year, count in counts if int(year) >= cutoff)
        print(sep.join(wordparts), str(count), sep=sep)
