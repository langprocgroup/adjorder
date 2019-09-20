#!/usr/bin/env Rscript
library(tidyverse)

args = commandArgs(trailingOnly=TRUE)
filename = args[1]

assert = stopifnot

d = read_csv(filename)
da = d %>%
   filter(deptype == "amod", num_deps == 0, left == "left", h_pos %in% c("NN", "NNS")) %>%
   mutate(logZ = log(sum(count)),
          logp_h_d = log(count) - logZ) %>%
   group_by(h_word) %>%
     mutate(logp_d = log(sum(count)) - logZ) %>%
     ungroup() %>%
   group_by(d_word) %>%
     mutate(logp_h = log(sum(count)) - logZ) %>%
     ungroup() %>%
   mutate(pmi=logp_h_d - logp_h - logp_d)

# Average must be positive
assert(with(da, sum(exp(logp_h_d) * pmi)) >= 0)

cat(format_csv(da))
     

   