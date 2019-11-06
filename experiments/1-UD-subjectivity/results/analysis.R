library(ggplot2)
library(lme4)
library(hydroGOF)
library(tidyr)
library(dplyr)

setwd("~/Desktop/adjs!/kids-adjectives/experiments/1-kids-subjectivity/Submiterator-master")
#setwd("~/git/kids-adjectives/experiments/1-kids-subjectivity/Submiterator-master/")


num_round_dirs = 12
df = do.call(rbind, lapply(1:num_round_dirs, function(i) {
  return (read.csv(paste(
    'round', i, '/kids-subjectivity.csv', sep='')) %>% #'round1/kids-subjectivity.csv')) %>% #for just 1
      mutate(workerid = (workerid + (i-1)*9)))}))

d = subset(df, select=c("workerid", "class","predicate","slide_number","response","language"))
unique(d$language)

length(unique(d$workerid)) # n=108
head(d)

## remove non-English speakers
d = d[d$language!="Russian"&d$language!="",]
length(unique(d$workerid)) # n=106

## determine number of observations
table(d$predicate)

## calculate average subjectivity by predicate
agg_adj = aggregate(response~predicate*class,data=d,mean)

## calculate average subjectivity by class
agg_class = aggregate(response~class,data=d,mean)

## write to CSV files
#write.csv(agg_adj,"../results/adjective-subjectivity.csv")
#write.csv(agg_class,"../results/class-subjectivity.csv")

