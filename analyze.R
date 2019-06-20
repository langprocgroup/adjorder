library(tidyverse)
##library(purrrlyr)
library(lme4)
library(lmtest)

assert = stopifnot

weighted_mean = function(ws, xs) {
    sum(ws*xs) / sum(ws)
}

replicate_row = function(r,n) {
    r[rep(1:nrow(r),n),1:ncol(r)]
}

unroll = function(d) {
    replicate_row(d, d$count)
}

getSubjRatings = function() {
  ## Michael Hahn wrote this function
  dataSubj1 = read.table("subjectivity-trials.csv",header=TRUE,sep=",")
  dataSubj1$X = 1
  dataSubj1$language = "UNK"
  dataSubj2 = read.table("subjectivity-expanded_results.csv",header=TRUE,sep=",")
  dataSubj =rbind(dataSubj1, dataSubj2)
  dataSubj = aggregate(dataSubj["response"],by=c(dataSubj["predicate"], dataSubj["class"]), mean, na.rm=TRUE)
  return(dataSubj)
}

subj = getSubjRatings() %>%
    rename(subjectivity=response,
           a=predicate)
pmi = read.csv("pmi.tsv", sep="\t", header=F)
names(pmi) = c("n", "a", "pmi")
pmi = pmi %>%
    separate(n, into=c("n", "pos")) %>%
    select(-pos) %>%
    separate(a, into=c("a", "pos")) %>%
    select(-pos) %>%
    group_by(n, a) %>%
      summarise(pmi=first(pmi)) %>%  # just take the first one
      ungroup()

aan = read.csv("aan.txt.gz", sep=" ", header=F)
assert(all(is.na(aan$V4)))
names(aan) = c("a1", "a2", "n", "trash", "count")
aan = aan %>% select(-trash)

an = read.csv("an.tsv", sep="\t", header=F)

d = aan %>%
    separate(a1, into=c("a1", "pos", "deleteme", "junk"), sep="/") %>%
      select(-deleteme, -junk, -pos) %>%
    separate(a2, into=c("a2", "pos", "deleteme", "junk"), sep="/") %>%
      select(-deleteme, -junk, -pos) %>%
    separate(n, into=c("n", "pos", "deleteme", "junk"), sep="/") %>%
      select(-deleteme, -junk, -pos) %>%
    group_by(a1, a2, n) %>%
      summarise(count=sum(count)) %>%
      ungroup() %>%
    filter(a1 != "", a2 != "", n != "")

anc = an %>%
    separate(V1, into=c("n", "junk"), sep="/") %>%
    select(-junk) %>%
    separate(V2, into=c("a", "junk"), sep="/") %>%
    select(-junk) %>%
    rename(count=V3) %>%
    inner_join(pmi)

mi = anc %>%
    group_by(a) %>%
      mutate(Z_a = sum(count),
             p_n_given_a = count / Z_a) %>%
      summarise(mi=sum(p_n_given_a * pmi)) %>%
    ungroup()

misubj = inner_join(mi, subj) 

ggplot(misubj, aes(x=mi, y=subjectivity, label=a)) +
    geom_text() +
    stat_smooth(method='lm') +
    theme_bw() +
    ylab("Subjectivity score") +
    xlab("Average pmi with noun")

ggsave("misubj.pdf", width=5, height=5)

with(misubj, cor.test(mi, subjectivity))
            
    
d2 = d %>%
    mutate(alphabetical_first=a1<a2,
           a_1=ifelse(alphabetical_first, a1, a2),
           a_2=ifelse(alphabetical_first, a2, a1)) %>%
    select(-a1, -a2) %>%
    rename(a1=a_1, a2=a_2) %>%
    inner_join(rename(pmi, a1=a, pmi1=pmi)) %>%
      inner_join(rename(pmi, a2=a, pmi2=pmi)) %>%
      mutate(pmi_diff=pmi2-pmi1) %>%
    inner_join(rename(subj, a1=a, subjectivity1=subjectivity)) %>%
      inner_join(rename(subj, a2=a, subjectivity2=subjectivity)) %>%
      mutate(subjectivity_diff=subjectivity2-subjectivity1) %>%    
    inner_join(rename(mi, a1=a, mi1=mi)) %>%
      inner_join(rename(mi, a2=a, mi2=mi)) %>%
      mutate(mi_diff=mi2-mi1)

p_second = d2 %>%
    select(count, a1, a2) %>%
    gather(position, a, -count) %>%
    separate(position, into=c("junk", "position"), sep=1) %>%
    select(-junk) %>%
    mutate(position=as.numeric(position)-1) %>%
    group_by(a) %>%
      summarise(p_second=sum(count*position)/sum(count)) %>%
      ungroup() %>%
    inner_join(mi) %>%
    inner_join(subj)

## Type-level regressions

## Do a train-test split

## d2 is 16681 rows, so hold out 10% = 1668

set.seed(0)

d2$hold_out = F
d2 = sample_frac(d2, 1) # Shuffle it
d2[1:1668,]$hold_out = T

d2_train = d2 %>% filter(!hold_out)
d2_test = d2 %>% filter(hold_out)
   
d2r = unroll(d2)
d2r_train = unroll(d2_train)
d2r_test = unroll(d2_test)

## Token regressions
ms_mr = d2r_train %>%
    mutate(pmi_diff.r=residuals(lm(pmi_diff ~ subjectivity_diff, data=.))) %>%
    glm(alphabetical_first ~ subjectivity_diff + pmi_diff.r, family="binomial", data=.)
ms_sr = d2r_train %>%
    mutate(subjectivity_diff.r=residuals(lm(subjectivity_diff ~ pmi_diff, data=.))) %>%
    glm(alphabetical_first ~ subjectivity_diff.r + pmi_diff, family="binomial", data=.)
s = d2r_train %>% glm(alphabetical_first~subjectivity_diff, family="binomial", data=.)
m = d2r_train %>% glm(alphabetical_first~pmi_diff, family="binomial", data=.)
ms = glm(alphabetical_first ~ subjectivity_diff + pmi_diff, family="binomial", data=d2r_train)

print(nrow(d2r_train))
print(nrow(d2r_test))


## Evaluate on the held-out tokens
predictions_token = d2r_test %>%
    mutate(pred_ms=predict(ms,.)>0, pred_s=predict(s,.)>0, pred_m=predict(m,.)>0,
           acc_ms=pred_ms==alphabetical_first,
           acc_s=pred_s==alphabetical_first,
           acc_m=pred_m==alphabetical_first)

## accuracies
predictions_token %>%
    summarise(m_ms=mean(acc_ms), m_s=mean(acc_s), m_m=mean(acc_m)) %>%
    print()

lrtest(s, ms)


## a sort of venn diagram of accuracies...
predictions_token %>%
    group_by(acc_both_s_m=acc_s & acc_m,
             acc_s_not_m=acc_s & !acc_m,
             acc_m_not_s=acc_m & !acc_s) %>%
      summarise(n=n()) %>%
    ungroup()
## both get 1037153 right
## MI exclusively gets 183968 right
## subjectivity exclusively gets 199182 right
## neither get 183982 right


## including the combined model...
predictions %>%
    group_by(acc_ms_only=acc_ms & !acc_m & !acc_s,
             bad_ms_only=!acc_ms & acc_m & acc_s) %>%
      summarise(n=n()) %>%
      ungroup()

## 322 things gotten by the combined model but not by m&s (all "large thick")
## 0 things gotten by m&s not gotten by the combined model




## A tibble: 4 × 4
##   acc_both_s_m acc_s_not_m acc_m_not_s       n
##          <lgl>       <lgl>       <lgl>   <int>
## 1        FALSE       FALSE       FALSE  146170
## 2        FALSE       FALSE        TRUE  189942
## 3        FALSE        TRUE       FALSE  207664
## 4         TRUE       FALSE       FALSE 1060509

## Cases where pmi improves on subjectivity:
cases = predictions %>% filter(acc_ms & !acc_s)

## They appear to be cases within categories.
## big long chains; black red flames; creamy smooth surface; different potential benefits; large long barrow; long thin features; young fresh roots; smooth hard wood; sweet spicy dishes; tiny short legs;

unpredictable = predictions %>% filter(!acc_ms & !acc_s & !acc_m)
## new young partner; smooth soft appearance; 
