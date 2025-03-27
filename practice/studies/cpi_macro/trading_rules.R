library(arrow)
library(data.table)
library(ggplot2)

# read data -----------------------------------------------------------------------------------

dfx <- as.data.table(read_parquet("data/workspace/dfX.pq"))
dfy <- as.data.table(read_parquet("data/workspace/dfY.pq"))

dftotrtn <- as.data.table(read_parquet("data/workspace/dftotrtn.pq"))
dftotrtn$date <- dftotrtn$`__index_level_0__`
dftotrtn$`__index_level_0__` <- NULL

# dfx <- dfx[, .(refmonthyear, cpiyoy, pmnmi, pmpmi, cpiyoy.surprise)]
# dfx <- na.omit(dfx)

# Examine data for trading rules --------------------------------------------------------------
# add thresholds for each value
pmi.thresh <- 50
cpi.thresh <- median(na.omit(dfx$cpiyoy))
dfx[, pmi.above := as.integer(pmpmi>pmi.thresh)]
dfx[, nmi.above := as.integer(pmnmi>pmi.thresh)]
dfx[, cpi.above := as.integer(cpiyoy>cpi.thresh)]

# Set *.above cols to -1 for short positions
# dfx[pmi.above==0, pmi.above:=-1]
# dfx[nmi.above==0, nmi.above:=-1]
# dfx[cpi.above==0, cpi.above:=-1]

# Set *.above cols to 0 for short positions, to test being out of market
# dfx[pmi.above==0, pmi.above:=-1]
# dfx[nmi.above==0, nmi.above:=-1]
# dfx[cpi.above==0, cpi.above:=-1]


# Try trading rule  ---------------------------------------------------------------------------
setindex(dfx, "refmonthyear")
setindex(dfy, "refmonthyear")

dtall <- data.table::merge.data.table(x=dfx, y=dfy, by="refmonthyear")

trade_by_rule <- function(signal, returns){
  r1 = cumprod(1.0 + returns) * 100
  sigrtn = cumprod(1.0 + (returns * signal)) * 100
  
  dt <- data.table()
  dt[, returns:= r1]
  dt[, sigreturns:= sigrtn]
  
  # colnames(dt) <- c(rtnlabel, siglabel)
  plot(dt$returns, type='l', col='red', lwd=3)
  lines(dt$sigreturns, type='l', col='black', lwd=3)
  
  return(dt)
}


idx6040 <- data.table()
idx6040[, returns:= dtall$`6040`]
idx6040[, sig.pmi:= dtall$pmi.above]
idx6040[, sig.nmi:= dtall$nmi.above]
idx6040[, sig.cpi:= dtall$cpi.above]

idx6040[, rtn.pmi := sig.pmi * returns]
idx6040[, rtn.nmi := sig.nmi * returns]
idx6040[, rtn.cpi := sig.cpi * returns]


idx6040[, .(mean.rtn = mean(returns),
            sd.rtn = sd(returns),
            mean.pmi.rtn = mean(rtn.pmi),
            sd.pmi.rtn = sd(rtn.pmi),
            mean.nmi.rtn = mean(rtn.nmi),
            sd.nmi.rtn = sd(rtn.nmi),
            mean.cpi.rtn = mean(rtn.cpi),
            mean.nmi.rtn = sd(rtn.cpi))]


idx6040[, .(mean.rtn = round(100*mean(returns), 2),
            sd.rtn = round(100*sd(returns), 2)),
        by = .(sig.pmi)]

idx6040[, .(mean.rtn = round(100*mean(returns), 2),
            sd.rtn = round(100*sd(returns), 2)),
        by = .(sig.nmi)]

idx6040[, .(mean.rtn = round(100*mean(returns), 2),
            sd.rtn = round(100*sd(returns), 2)),
        by = .(sig.cpi)]



# Inspect returns by signal -------------------------------------------------------------------
# "spy"             "agg"             "xop"             "xme"             "xhb"             "gld"            
# [15] "bil"             "tip"             "spts"            "spti"            "naivevol"        "6040"         

wgt.pmi = -.25
wgt.nmi = 1.5
wgt.cpi = -.25
dtall[, comboSignal:=(pmi.above*wgt.pmi + nmi.above*wgt.nmi + cpi.above*wgt.cpi)]
sigcol <- "comboSignal"

rtncol <- "spy"
strat.returns <- trade_by_rule(dtall[[sigcol]], returns=dtall[[rtncol]])



# How many months have pos returns ------------------------------------------------------------
poscount <- function(monthlyReturns){
  return(sum(monthlyReturns>0))
}
round(c(poscount(dfy$spy), poscount(dfy$agg), poscount(dfy$xop), 
  poscount(dfy$xme), poscount(dfy$xhb), poscount(dfy$gld), 
  poscount(dfy$bil), poscount(dfy$tip), poscount(dfy$spts), 
  poscount(dfy$spti), poscount(na.omit(dfy$naivevol)), poscount(dfy$`6040`))/dim(dfy)[1], 2)

dim(dfy)

head(dfy)


