
# imports -------------------------------------------------------------------------------------
library(arrow)
library(data.table)
library(ggplot2)
library(data.table)
library(clipr)
library(quantmod)
# load data -----------------------------------------------------------------------------------

df <- as.data.table(read_parquet("data/results/results.pq"))
dt <- as.data.table(df)

vols <- dt[, .(basevol=round(sd(y_rtn) * sqrt(12), 4) * 100,
       lsvol=round(sd(strat_ls_rtn) * sqrt(12), 4) * 100,
       lovol=round(sd(strat_lo_rtn) *sqrt(12), 4) * 100), by=ticker]

trrs <- dt[, 
           .(basetr=cumprod(1+y_rtn)*100,
             lotr=cumprod(1+strat_lo_rtn)*100,
             lstr=cumprod(1+strat_ls_rtn)*100), 
           by=ticker]

trrs <- trrs[, 
             .(basetr=round(last(basetr), 2), 
               lotr=round(last(lotr), 2), 
               lstr=round(last(lstr), 2)), 
             by=ticker]


setkey(trrs, ticker)
setkey(vols, ticker)

results <- cbind(trrs, vols)

clipr::write_clip(results)

# ref data ------------------------------------------------------------------------------------

ref <- fread("data/syms.csv", header=T)
ref$V1 <- NULL

colnames(ref) <- c('ticker', 'name', 'launchdate', 'group', 'subgroup', 'style', 'maturity', 'duration')
setkey(ref, ticker)

clipr::write_clip(ref)
