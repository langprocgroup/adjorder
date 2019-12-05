library(ggplot2)
library(lme4)
library(hydroGOF)
library(tidyr)
library(dplyr)

setwd("~/git/adjorder/experiments/1-UD-subjectivity/Submiterator-master")
#setwd("~/git/kids-adjectives/experiments/1-kids-subjectivity/Submiterator-master/")

df = read.csv("1-UD-subjectivity-trials.csv",header=T)
s = read.csv("1-UD-subjectivity-subject_information.csv",header=T)
d = subset(df, select=c("workerid", "class","predicate","slide_number","response"))
d$language = s$language[match(d$workerid,s$workerid)]
d$assess = s$asses[match(d$workerid,s$workerid)]
d$age = s$age[match(d$workerid,s$workerid)]
d$gender = s$gender[match(d$workerid,s$workerid)]
unique(d$language)

length(unique(d$workerid)) # n=280
head(d)

## remove non-English speakers
d = d[d$language=="english"|
        d$language=="English"|
        d$language=="English "|
        d$language=="ENGLISH"|
        d$language=="englsh"|
        d$language=="enlish"|
        d$language=="Englisha"|
        d$language=="eNGLISH"|
        d$language=="Englsh"
        ,]
length(unique(d$workerid)) # n=264

## load helper file for bootstrapped CIs
source("../results/helpers.r")

agr = bootsSummary(data=d, measurevar="response", groupvars=c("predicate"))

mean(agr$N) ## 19.8995

## write to CSV files
#write.csv(agr,"../results/adjective-subjectivity.csv")
#write.csv(agg_class,"../results/class-subjectivity.csv")

