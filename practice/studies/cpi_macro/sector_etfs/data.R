require(quantmod)
require(data.table)
require(ggplot2)
require(tidyverse)

# data fetch ------------------------------------------------------------------------



tickers = c(
  "XLC",  # Communication Services
  "XLY",  # Consumer Discretionary
  "XLP",  # Consumer Staples
  "XLE",  # Energy
  "XLF",  # Financials
  "XLV",  # Health Care
  "XLI",  # Industrials
  "XLB",  # Materials
  "XLRE",  # Real Estate
  "XLU",  # Utilities
  "SPY" # broad market index
)

getSymbols(tickers, from="2000-01-01")


# data munging ----------------------------------------------------------------------


chartSeries(XLC)
head(XLE)
