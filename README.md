# Dual_Mom Reasearch Code
This research aims to enhance the long-only Dual Momentum strategy by integrating volatility as a market shift indicator and implementing a dynamic asset allocation framework across multiple asset classes.
Therefore, the code are breakdown into three parts, and each folder are a summary for the work under each part. Noting that all our codes are implemented under Google Colab, thus there are some libraies we used are limited to colab environment; but for replication, you can easily find similar replcament under any python enviornment.

Here are the breakdowns and steps for our research:

1. Replcation and Performance Analysis: (replicate&analysis folder)
- Replicate file are responsible for replicating long-only Dual mom strategy across four asset classes: equities, bonds, commodities, and REITs. It requires monthly return data for all the risky and riskless assets mentioned in our thesis. In the end, it will return a file with the return of dual_mom, relative_mom, abs_mom strategy across for asset classes.
- Factor Anaysis file will requires the input of return data from Replicate file. In addition, this require the factor data from FF website. This file will conduct factor analysis, as well as in-sample and out-sample robustness tests.

2. Regime Siganl Models: (regime_siganl folder)\
- 
