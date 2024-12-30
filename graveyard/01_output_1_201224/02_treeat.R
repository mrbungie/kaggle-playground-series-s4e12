
install.packages("arrow")
install.packages("missForest")

library(arrow)
library(missForest)

df <- read_parquet("data.parquet")
