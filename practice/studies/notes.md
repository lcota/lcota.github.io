Thoughts
Risk Parity portfolios are meant to deal with shocks to expectation. Surprises to GDP (ISM) and Inflation (CPI) lead to over/under performance of Stocks/Bonds. 



Next Steps:
    * Add Survey estimates to CPI, ISM Manufacturing, ISM Non Manufacturing
    


Done Tasks
*TODO: Create short-hand notation for column names*
*Add publish_date col to data set*
*Pick ETF list for US inflation data*
*Add SPY/VOO/ sector etfs, small/mid/large cap etf*
* Add GLD (commods) *

1. What is the Consensus Forecast?
2. How is the Realized k-month interpreted? Is this the realized inflation of goods 12 months later? For example, if we use 31-Jan-2018 as a reference point, the data date (publish date) for this is 14-Feb-2018. 

ref date    pub date    horizon     column              value
1/31/2018   2/14/2018   2/28/2018   1M YoY forecast     2.10
1/31/2018   2/14/2018   1/31/2019   12M YoY forecast    1.62
1/31/2018   2/14/2018               CPI Realised        2.07

-> k-month realized is the BLS realized value 12 months later. 


At each time t, Turnleaf generates forecasts for next period inflation through 12mth inflation. 

3. What is the data publish date for a given row? 


