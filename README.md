# Dual_Mom Reasearch Code
This research aims to enhance the long-only Dual Momentum strategy by integrating volatility as a market shift indicator and implementing a dynamic asset allocation framework across multiple asset classes.
Therefore, the code are breakdown into three parts, and each folder are a summary for the work under each part. Noting that all our codes are implemented under Google Colab, thus there are some libraies we used are limited to colab environment; but for replication, you can easily find similar replcament under any python enviornment.

Here are the breakdowns and steps for our research:

1. Replcation and Performance Analysis: (replicate&analysis folder)
- Replicate file are responsible for replicating long-only Dual mom strategy across four asset classes: equities, bonds, commodities, and REITs. It requires monthly return data for all the risky and riskless assets mentioned in our thesis. In the end, it will return a file with the return of dual_mom, relative_mom, abs_mom strategy across for asset classes.
- Factor Anaysis file will requires the input of return data from Replicate file. In addition, this require the factor data from FF website. This file will conduct factor analysis, as well as in-sample and out-sample robustness tests.

2. Regime Signal Models: (regime_signal folder)
- create_Regime_data file requires the regime data input specified in our thesis. This is mainly responisble for data cleaning, VRP calculation and eventual merging.
- 1_regime_Y_Selection file use the output from previous file, and conduct selection on the eventual Y selection for out regime model. Spoilers alert, our research found VIX to be the answer.
- 2_Final Regime are used to determine the threshold level following regime selection. This serves as a robustness test for the eventual threshold level.
- 3_Final_model file conducts model selection to determine the eventual regime model we will applied
- 4_regime_performance: after select final model, we learned how dual mom strategy performed under different regimes, which provides insights for the eventual integration of the regime model and our dynamic asset allocation framework.

3. Dynamic Asset Allocation: (DAA folder)
- allocation.py is responsible for creating all types of allocation schema and comparing their performances. This file take the return data output from "Replicate" and regime data output from "3_Final_model". 
