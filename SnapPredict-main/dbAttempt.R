install.packages("nflfastR")
install.packages("nflreadr")  # optional: for fast loading
library(nflfastR)
library(nflreadr)
library(dplyr)  # useful for data manipulation

setwd("/Users/siddharthsrinivasan/gt_class/yr3/cs4641/") 


pbp <- nflfastR::load_pbp(2019:2022)

write.csv(pbp, file = "nfl_pbp_2019_2022.csv", row.names = FALSE)

file.exists("nfl_pbp_2019_2022.csv")

getwd()