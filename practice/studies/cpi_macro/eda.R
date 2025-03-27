# Imports -----------------------------------------------------------------
library(arrow)
library(data.table)
library(ggplot2)
library(lubridate)
library(feather)
library(stringr)
library(quantmod)

# CPI / ISM Fred Data
# US CPI: CPIAUCSL
quantmod::getSymbols.FRED(Symbols=c('CPIAUCNS'), env=.GlobalEnv)
# quantmod::getSymbols(Symbols=c("^GSPC"), env=.GlobalEnv, from=strftime("1947-01-01"))
getSymbols('SPY', from=strftime("1947-01-01"))

# save data to disk ---------------------------------------------------------------------------
cpi <- data.table(CPIAUCNS)
cpi$date <- index(CPIAUCNS)
cpi <- cpi[, .(date, CPIAUCNS)]
colnames(cpi) <- c('date', 'usa.cpi')

spy <- data.table(SPY)
spy$date <- index(SPY)
colnames(spy) <- c('open', 'high', 'low', 'close', 'volume', 'adjusted', 'date')
spy <- spy[, .(date, open, high, low, close, volume, adjusted)]

agg <- data.table(AGG)
agg$date <- index(AGG)
colnames(agg) <- c('open', 'high', 'low', 'close', 'volume', 'adjusted', 'date')
agg <- agg[, .(date, open, high, low, close, volume, adjusted)]

fwrite(cpi, file="c:/users/lcota/dropbox/prj/Turnleaf_Analytics/data/raw/cpi_fred.csv", append=F)
fwrite(spy, file="c:/users/lcota/dropbox/prj/Turnleaf_Analytics/data/raw/spy_all.csv", append=F)
fwrite(agg, file="c:/users/lcota/dropbox/prj/Turnleaf_Analytics/data/raw/agg_all.csv", append=F)
# sp500 <- data.table(GSPC$GSPC.Adjusted)

# cpi <- data.table(CPIAUCSL)
# load data ---------------------------------------------------------------

ts <- merge.xts(cpi, sp500[index(cpi)], join="left")
ts <- na.omit(ts)

# plots ---------------------------------------------------------------------------------------

plot.xts(ts$CPIAUCSL)
par(new=T)
plot.xts(ts$GSPC.Adjusted, yaxis.same=F)

plot(cpi)
addSeries(sp500$GSPC.Adjusted, on=1)


# write data to file --------------------------------------------------------------------------
write.csv(cpi, "c:/users/lcota/dropbox/prj/Turnleaf_Analytics/data/raw/cpi_fred.csv")


