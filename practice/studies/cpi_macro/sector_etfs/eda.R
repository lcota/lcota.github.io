# Examine time series relationship of Inflation changes & forward returns
library(arrow)
library(data.table)
df = arrow::read_parquet("workspace/cpi_fred_factors.pq")
