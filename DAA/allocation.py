# -*- coding: utf-8 -*-
"""
Refined script for analyzing Dual Momentum strategies with MVO enhancements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# import statsmodels.api as sm # Not used in the provided strategy functions
import warnings
import os # For checking file paths and saving figures

# --- Configuration ---

# Suppress warnings (use cautiously)
warnings.filterwarnings('ignore')

# Strategy Parameters
TBILL_COL       = 'TB3M_ROR'          # Column name for the risk-free rate (T-Bill)
LOOKBACK_MONTHS = 120                 # Rolling window lookback period in months
MIN_LOOKBACK    = 12                  # Minimum months required for estimation
START_DATE      = '2000-01-01'        # Common start date for strategy evaluation
REGIME_SIGNAL_PRED = 'predicted_VIX_threshold' # Column for predicted regime (0 or 1)
REGIME_SIGNAL_REAL = 'real_VIX_threshold'      # Column for actual regime (used for estimation)

# Asset Universe for MVO Strategies (excluding MVO_Dual_Equity)
DM_SERIES = ['Dual_Equity', 'Dual_Credit', 'Dual_REIT', 'Dual_Stress']

# Plotting & Saving Options
SAVE_FIGS  = True  # Set to True to save figures
FIG_PATH   = './'  # Directory to save figures (current directory)
FIG_DPI    = 300   # Resolution for saved figures

# --- Data Loading ---

def load_data(path_returns='returns.csv', path_regime='regime_signal.csv'):
    """
    Loads returns and regime signal data, merges them, and sets up the DataFrame.

    Args:
        path_returns (str): Path to the CSV file containing asset returns.
                            Expected columns: 'Date', DM_SERIES, TBILL_COL.
        path_regime (str): Path to the CSV file containing regime signals.
                           Expected columns: 'Date', REGIME_SIGNAL_PRED, REGIME_SIGNAL_REAL.

    Returns:
        pandas.DataFrame or None: Merged DataFrame indexed by Date, or None if loading fails.
    """
    print(f"Loading returns data from: {path_returns}")
    if not os.path.exists(path_returns):
        print(f"Error: Returns file not found at '{path_returns}'")
        return None
    try:
        returns = pd.read_csv(path_returns, parse_dates=['Date'])
        returns = returns.sort_values('Date').ffill().bfill() # Sort and fill missing values
    except Exception as e:
        print(f"Error reading returns file: {e}")
        return None

    print(f"Loading regime data from: {path_regime}")
    if not os.path.exists(path_regime):
        print(f"Error: Regime file not found at '{path_regime}'")
        return None
    try:
        regime = pd.read_csv(path_regime, parse_dates=['Date'])
        # Align regime dates to the start of the month for consistent merging
        regime['Date'] = regime['Date'].values.astype('datetime64[M]')
    except Exception as e:
        print(f"Error reading regime file: {e}")
        return None

    # Merge returns and regime data
    print("Merging data...")
    df = pd.merge(returns, regime, on='Date', how='inner')

    # Filter data from the earliest regime signal date onwards
    start_date_data = regime['Date'].min()
    df = df[df['Date'] >= start_date_data].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Define required columns
    signal_cols = [REGIME_SIGNAL_PRED, REGIME_SIGNAL_REAL]
    required_cols = DM_SERIES + [TBILL_COL] + signal_cols
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Error: Missing required columns after merge: {missing_cols}")
        return None

    # Select necessary columns and set index
    keep_cols = ['Date'] + required_cols
    df = df[keep_cols].set_index('Date')
    print("Data loading and preparation complete.")
    return df

# --- Core Optimization and Metrics ---

def optimize_mvo(mu, cov, rf_rate, assets):
    """
    Performs Mean-Variance Optimization to find weights maximizing Sharpe ratio.

    Args:
        mu (pd.Series): Expected returns for the assets.
        cov (pd.DataFrame): Covariance matrix for the assets.
        rf_rate (float): Risk-free rate.
        assets (list): List of asset names corresponding to mu and cov.

    Returns:
        np.ndarray: Optimal asset weights, or equal weights if optimization fails.
    """
    n = len(assets)
    if n == 0:
        return np.array([])

    # Ensure consistency
    try:
        mu_ordered = mu.loc[assets]
        cov_ordered = cov.loc[assets, assets]
    except KeyError as e:
        print(f"Error: Asset mismatch in MVO inputs. Missing: {e}. Assets: {assets}. Mu index: {mu.index}. Cov index: {cov.index}")
        return np.ones(n) / n # Fallback

    # Check for NaNs/Infs before optimization
    if mu_ordered.isnull().any() or cov_ordered.isnull().any().any() or \
       np.isinf(mu_ordered).any() or np.isinf(cov_ordered).any().any() or pd.isna(rf_rate):
        print(f"Warning: NaN/inf found in MVO inputs for assets {assets}. Falling back to equal weights.")
        return np.ones(n) / n # Fallback

    w0 = np.ones(n) / n  # Initial guess: equal weights
    bounds = [(0, 1)] * n # Constraints: weights between 0 and 1 (long-only)

    def neg_sharpe(w):
        """Objective function: negative Sharpe ratio."""
        port_ret = np.dot(w, mu_ordered)
        # Ensure positive semi-definite covariance matrix calculation
        # Add small epsilon for numerical stability, ensure non-negative variance
        variance = w @ cov_ordered @ w
        vol = np.sqrt(max(variance, 1e-12)) # Use max to avoid negative sqrt

        if vol < 1e-9:
             return 0 # Avoid division by zero, Sharpe is undefined or zero
        return -((port_ret - rf_rate) / vol)

    # Constraint: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Optimization using SLSQP (suitable for constrained optimization)
    sol = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if sol.success and np.abs(np.sum(sol.x) - 1) < 1e-6 and np.all(sol.x >= -1e-6):
        # Normalize weights slightly if needed due to solver precision
        final_weights = np.maximum(sol.x, 0) # Ensure non-negative weights
        final_weights /= np.sum(final_weights)
        return final_weights
    else:
        # Fallback to equal weight if optimization fails or constraints violated
        # print(f"Warning: MVO optimization failed (Success: {sol.success}, Message: {sol.message}). Falling back to equal weights for assets: {assets}")
        return w0

def calc_metrics(returns, rf_rate_series):
    """
    Calculates standard performance metrics for a returns series.

    Args:
        returns (pd.Series): Time series of strategy returns.
        rf_rate_series (pd.Series): Time series of the risk-free rate.

    Returns:
        dict: Dictionary containing calculated performance metrics.
    """
    # Ensure inputs are pandas Series and handle NaNs
    y = returns.dropna()
    if y.empty:
        print("Warning: Empty returns series provided for metric calculation.")
        return {k: np.nan for k in ['Annual Return', 'Excess Return', 'Annual Std', 'Sharpe', 'Max Drawdown', '% Profit Months']}

    # Align risk-free rate to returns index
    rf_aligned = rf_rate_series.reindex(y.index).ffill().bfill()
    if rf_aligned.isnull().any():
         print("Warning: NaNs in aligned risk-free rate series.")
         # Decide handling: dropna, fill with 0, or return NaNs
         rf_aligned = rf_aligned.fillna(0) # Example: fill with 0

    excess = y - rf_aligned # Use simple subtraction for monthly excess returns

    if len(y) < MIN_LOOKBACK:
         print(f"Warning: Less than {MIN_LOOKBACK} non-NaN data points for metric calculation. Annualized results may be unreliable.")

    # Calculate Metrics
    months = len(y)
    annual_factor = 12

    # Geometric Annualized Return
    geometric_ann_ret = (1 + y).prod()**(annual_factor / months) - 1

    # Arithmetic Annualized Excess Return (for Sharpe)
    # Note: Using arithmetic mean for Sharpe is standard, geometric for cumulative wealth
    arithmetic_ann_exc_ret = excess.mean() * annual_factor

    # Annualized Volatility (Standard Deviation of monthly returns)
    ann_std = y.std() * np.sqrt(annual_factor)

    # Sharpe Ratio (using arithmetic excess return)
    sharpe = arithmetic_ann_exc_ret / ann_std if ann_std > 1e-9 else np.nan

    # Max Drawdown
    cum = (1 + y).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1
    mdd = dd.min() if not dd.empty else np.nan

    # % Profitable Months
    win_rate = (y > 0).mean()

    return {
        'Annual Return': geometric_ann_ret, # Report geometric return
        'Excess Return': arithmetic_ann_exc_ret, # Arithmetic for Sharpe
        'Annual Std': ann_std,
        'Sharpe': sharpe,
        'Max Drawdown': mdd,
        '% Profit Months': win_rate
    }

# --- Parameter Estimation Helper Functions ---

def estimate_params(data_window, assets, rf_col):
    """Estimates mu, cov, rf from a data window."""
    if data_window.empty or len(data_window) < MIN_LOOKBACK:
        return None, None, None
    if not all(c in data_window.columns for c in assets + [rf_col]):
        return None, None, None

    mu = data_window[assets].mean()
    cov = data_window[assets].cov()
    rf = data_window[rf_col].mean()

    # Basic check for validity
    if mu.isnull().any() or cov.isnull().any().any() or pd.isna(rf) or \
       np.isinf(mu).any() or np.isinf(cov).any().any():
        return None, None, None

    return mu, cov, rf

def estimate_params_regime(hist_data, current_pred_regime, regime_col, assets, rf_col):
    """Estimates parameters using regime-filtered expanding window data."""
    if hist_data.empty or len(hist_data) < MIN_LOOKBACK:
        return None, None, None

    regime_data = hist_data[hist_data[regime_col] == current_pred_regime]

    # Fallback 1: Use full history if regime subset is too small
    if len(regime_data) < MIN_LOOKBACK:
        print(f"Warning: Not enough data ({len(regime_data)}) in regime {current_pred_regime}. Using full history for estimation.")
        regime_data = hist_data # Use full history

    # Fallback 2: Return None if even full history is too small
    if len(regime_data) < MIN_LOOKBACK:
        return None, None, None

    return estimate_params(regime_data, assets, rf_col)


def estimate_params_regime_rolling(rolling_window, full_hist_before_current, current_pred_regime, regime_col, assets, rf_col, lookback):
    """Estimates parameters using regime-filtered rolling window data with fallbacks."""
    if rolling_window.empty or len(rolling_window) < MIN_LOOKBACK:
        return None, None, None

    regime_data = rolling_window[rolling_window[regime_col] == current_pred_regime]

    # Fallback logic if regime subset in window is too small
    if len(regime_data) < MIN_LOOKBACK:
        print(f"Warning: Not enough data ({len(regime_data)}) in rolling regime {current_pred_regime}. Trying full history regime subset.")
        # Try full history for that regime, before current date
        full_hist_regime = full_hist_before_current[full_hist_before_current[regime_col] == current_pred_regime]

        if len(full_hist_regime) >= MIN_LOOKBACK:
             # Take last 'lookback' periods if available, else take all available
             regime_data = full_hist_regime.iloc[-lookback:]
             print(f"Using fallback: last {len(regime_data)} periods from full history regime {current_pred_regime}.")
        else:
             print(f"Warning: Not enough data in full history regime {current_pred_regime}. Using full rolling window.")
             regime_data = rolling_window # Fallback to full window
             if len(regime_data) < MIN_LOOKBACK:
                 return None, None, None # Cannot estimate even with full window

    return estimate_params(regime_data, assets, rf_col)


# --- Generic Backtesting Engine (Optional Refactoring - complex to capture all nuances) ---
# Note: Due to the complexity and specific nuances (esp. Blended), keeping separate
# strategy functions calling helper estimators might be clearer for now.
# We will keep the separate functions but streamline them using the helpers where possible.


# --- Strategy Return Calculation Functions ---

def run_mvo_strategy(df, assets_to_opt, strategy_name, start_date,
                     param_estimator_func, rf_col=TBILL_COL, **estimator_kwargs):
    """
    Core loop for running MVO-based strategies.

    Args:
        df (pd.DataFrame): The main dataframe with returns and signals.
        assets_to_opt (list): Assets to include in the MVO.
        strategy_name (str): Name for the output Series.
        start_date (str): Start date for calculations.
        param_estimator_func (callable): Function to estimate mu, cov, rf.
        rf_col (str): Name of the risk-free rate column.
        **estimator_kwargs: Additional arguments for the estimator function.

    Returns:
        pd.Series: Strategy returns.
    """
    print(f"Running strategy: {strategy_name}...")
    ret = pd.Series(np.nan, index=df.index)
    start_dt = pd.to_datetime(start_date)

    for i, current_date in enumerate(df.index):
        if current_date < start_dt: continue

        # Pass necessary data slices and arguments to the specific estimator
        estimator_args = {'df': df, 'current_date_index': i, 'assets': assets_to_opt, 'rf_col': rf_col}
        estimator_args.update(estimator_kwargs) # Add strategy-specific args

        mu, cov, rf_rate = param_estimator_func(**estimator_args)

        if mu is None or cov is None or rf_rate is None:
            # print(f"Skipping {current_date} for {strategy_name} due to invalid parameters.")
            continue # Skip if parameters couldn't be estimated

        weights = optimize_mvo(mu, cov, rf_rate=rf_rate, assets=assets_to_opt)

        # Calculate return for the current period
        current_returns = df[assets_to_opt].iloc[i]
        if current_returns.isnull().any(): continue # Skip if current returns are NaN
        ret.iat[i] = np.dot(weights, current_returns.values)

    ret.name = strategy_name
    print(f"Finished strategy: {strategy_name}")
    return ret

# --- Specific Estimator Wrappers ---
# These functions prepare the data slices needed by the generic estimate_params helpers

def estimator_wrapper_mvo_dual_equity(df, current_date_index, assets, rf_col, lookback):
    """Provides data window for MVO Dual Equity."""
    window = df.iloc[current_date_index - lookback : current_date_index] \
             if current_date_index >= lookback else df.iloc[:current_date_index]
    return estimate_params(window, assets, rf_col)

def estimator_wrapper_full_mvo_expanding(df, current_date_index, assets, rf_col):
    """Provides data window for Full MVO Expanding."""
    hist = df.iloc[:current_date_index]
    return estimate_params(hist, assets, rf_col)

def estimator_wrapper_rolling_mvo_10yr(df, current_date_index, assets, rf_col, lookback):
    """Provides data window for Rolling MVO."""
    window = df.iloc[max(0, current_date_index - lookback) : current_date_index]
    return estimate_params(window, assets, rf_col)

def estimator_wrapper_regime_based_full_mvo(df, current_date_index, assets, rf_col):
    """Provides data window for Regime-Based Full MVO."""
    hist = df.iloc[:current_date_index]
    pred_regime = df[REGIME_SIGNAL_PRED].iloc[current_date_index]
    return estimate_params_regime(hist, pred_regime, REGIME_SIGNAL_REAL, assets, rf_col)

def estimator_wrapper_regime_based_rolling_mvo(df, current_date_index, assets, rf_col, lookback):
    """Provides data window for Regime-Based Rolling MVO."""
    rolling_window = df.iloc[max(0, current_date_index - lookback) : current_date_index]
    full_hist = df.iloc[:current_date_index]
    pred_regime = df[REGIME_SIGNAL_PRED].iloc[current_date_index]
    return estimate_params_regime_rolling(rolling_window, full_hist, pred_regime, REGIME_SIGNAL_REAL, assets, rf_col, lookback)

# --- Strategy Definitions using the Engine ---

def strategy_mvo_dual_equity_refactored(df):
    return run_mvo_strategy(df,
                            assets_to_opt = ['Dual_Equity', TBILL_COL],
                            strategy_name = 'MVO_Dual_Equity',
                            start_date    = START_DATE,
                            param_estimator_func = estimator_wrapper_mvo_dual_equity,
                            lookback = LOOKBACK_MONTHS)

def strategy_equal_weight_refactored(df):
    """Equal weight does not use MVO, so keep its simpler definition."""
    print("Running strategy: Equal_Weight...")
    df_sub = df[df.index >= pd.to_datetime(START_DATE)].copy()
    ret = df_sub[DM_SERIES].mean(axis=1)
    ret.name = 'Equal_Weight'
    print("Finished strategy: Equal_Weight")
    return ret

def strategy_full_mvo_expanding_refactored(df):
     return run_mvo_strategy(df,
                            assets_to_opt = DM_SERIES,
                            strategy_name = 'Full_MVO_Expanding',
                            start_date    = START_DATE,
                            param_estimator_func = estimator_wrapper_full_mvo_expanding)

def strategy_rolling_mvo_10yr_refactored(df):
     return run_mvo_strategy(df,
                            assets_to_opt = DM_SERIES,
                            strategy_name = 'Roll_MVO',
                            start_date    = START_DATE,
                            param_estimator_func = estimator_wrapper_rolling_mvo_10yr,
                            lookback = LOOKBACK_MONTHS)

def strategy_regime_based_full_mvo_refactored(df):
     return run_mvo_strategy(df,
                            assets_to_opt = DM_SERIES,
                            strategy_name = 'Regime_Based_Full_MVO',
                            start_date    = START_DATE,
                            param_estimator_func = estimator_wrapper_regime_based_full_mvo)

def strategy_regime_based_rolling_mvo_refactored(df):
     return run_mvo_strategy(df,
                            assets_to_opt = DM_SERIES,
                            strategy_name = 'Regime_Based_Rolling_MVO',
                            start_date    = START_DATE,
                            param_estimator_func = estimator_wrapper_regime_based_rolling_mvo,
                            lookback = LOOKBACK_MONTHS)

# Blended strategy still needs its own specific logic due to blending/shrinkage steps
def strategy_blended_regime_mvo_refactored(df, shrink_lambda=0.5):
    """Calculates returns for the Blended Regime MVO strategy."""
    print("Running strategy: Blended_Regime_MVO...")
    ret = pd.Series(np.nan, index=df.index)
    assets_to_opt = DM_SERIES
    start_dt = pd.to_datetime(START_DATE)

    for i, current_date in enumerate(df.index):
        if current_date < start_dt: continue

        hist = df.iloc[:i]
        if len(hist) < MIN_LOOKBACK: continue

        # --- Estimate params for regimes and full ---
        # Regime 0
        mu_0, cov_0, rf_0 = estimate_params_regime(hist, 0, REGIME_SIGNAL_REAL, assets_to_opt, TBILL_COL)
        if mu_0 is None: continue # Skip if estimation failed

        # Regime 1
        mu_1, cov_1, rf_1 = estimate_params_regime(hist, 1, REGIME_SIGNAL_REAL, assets_to_opt, TBILL_COL)
        if mu_1 is None: continue # Skip if estimation failed

        # Full History
        mu_full, cov_full, rf_full = estimate_params(hist, assets_to_opt, TBILL_COL)
        if mu_full is None: continue # Skip if estimation failed

        # --- Optimize for each component ---
        w0 = optimize_mvo(mu_0, cov_0, rf_rate=rf_0, assets=assets_to_opt)
        w1 = optimize_mvo(mu_1, cov_1, rf_rate=rf_1, assets=assets_to_opt)
        w_full = optimize_mvo(mu_full, cov_full, rf_rate=rf_full, assets=assets_to_opt)

        # --- Blend ---
        pred_regime_hist = df[REGIME_SIGNAL_PRED].iloc[max(0, i - 2): i + 1] # Use last 3 including current
        alpha = pred_regime_hist.mean() if len(pred_regime_hist) > 0 else 0
        alpha = min(max(alpha, 0), 1)

        w_blended = alpha * w1 + (1 - alpha) * w0

        # --- Shrink ---
        w_final = (1 - shrink_lambda) * w_blended + shrink_lambda * w_full
        w_final = w_final / np.sum(w_final) # Re-normalize

        # --- Calculate Return ---
        current_returns = df[assets_to_opt].iloc[i]
        if current_returns.isnull().any(): continue
        ret.iat[i] = np.dot(w_final, current_returns.values)

    ret.name = 'Blended_Regime_MVO'
    print("Finished strategy: Blended_Regime_MVO")
    return ret


# --- Main Execution ---

if __name__ == "__main__":

    # Load Data
    df = load_data()

    if df is not None:
        print("\n--- Calculating Strategy Returns ---")
        # Define strategies to run
        strategies = {
            "MVO_Dual_Equity": strategy_mvo_dual_equity_refactored,
            "Equal_Weight": strategy_equal_weight_refactored,
            "Full_MVO_Expanding": strategy_full_mvo_expanding_refactored,
            "Roll_MVO": strategy_rolling_mvo_10yr_refactored,
            "Regime_Based_Full_MVO": strategy_regime_based_full_mvo_refactored,
            "Regime_Based_Rolling_MVO": strategy_regime_based_rolling_mvo_refactored,
            "Blended_Regime_MVO": strategy_blended_regime_mvo_refactored,
        }

        # Calculate returns
        all_returns = {name: func(df) for name, func in strategies.items()}
        combined_returns = pd.concat(all_returns.values(), axis=1, keys=all_returns.keys())

        # Filter returns dataframe from the specified start date AFTER concatenation
        combined_returns = combined_returns[combined_returns.index >= pd.to_datetime(START_DATE)]

        # Drop columns that are entirely NaN (if a strategy failed completely)
        combined_returns.dropna(axis=1, how='all', inplace=True)

        if combined_returns.empty:
            print("\nError: No strategy produced valid returns.")
        else:
            print("\n--- Combined Returns Head ---")
            print(combined_returns.head())
            print("\n--- Combined Returns Info ---")
            combined_returns.info()

            # --- Calculate Metrics ---
            print("\n--- Calculating Performance Metrics ---")
            rf_series = df[TBILL_COL]
            metrics = {}
            for col in combined_returns.columns:
                strategy_ret = combined_returns[col].dropna()
                if not strategy_ret.empty:
                    metrics[col] = calc_metrics(strategy_ret, rf_series)
                else:
                    metrics[col] = {k: np.nan for k in ['Annual Return', 'Excess Return', 'Annual Std', 'Sharpe', 'Max Drawdown', '% Profit Months']}

            metrics_df = pd.DataFrame(metrics).T.sort_values('Sharpe', ascending=False) # Sort by Sharpe
            print("\n--- Performance Metrics ---")
            # Format for better readability
            metrics_df_display = metrics_df.copy()
            for col in ['Annual Return', 'Excess Return', 'Annual Std', 'Max Drawdown', '% Profit Months']:
                metrics_df_display[col] = metrics_df_display[col].map('{:.2%}'.format)
            metrics_df_display['Sharpe'] = metrics_df_display['Sharpe'].map('{:.2f}'.format)
            print(metrics_df_display.to_string())

            # --- Plotting Section ---
            print("\n--- Generating Plots ---")

            # Ensure FIG_PATH exists
            if SAVE_FIGS and not os.path.exists(FIG_PATH):
                os.makedirs(FIG_PATH)
                print(f"Created directory: {FIG_PATH}")

            # Plot 1: Cumulative Return
            print("Plotting Cumulative Returns...")
            plt.figure(figsize=(14, 7)) # Wider figure
            cumulative_returns = (1 + combined_returns.dropna(how='all')).cumprod()
            # Normalize to start at 1 correctly, handling potential initial NaNs
            first_valid_idx = cumulative_returns.first_valid_index()
            if first_valid_idx is not None:
                 cumulative_returns_norm = cumulative_returns / cumulative_returns.loc[first_valid_idx]
                 for col in cumulative_returns_norm.columns:
                     label=col.replace('_', ' ')
                     linewidth = 3 if col == 'Blended_Regime_MVO' else (2 if col == 'Full_MVO_Expanding' else 1.5)
                     alpha = 1 if col in ['Blended_Regime_MVO', 'Full_MVO_Expanding'] else 0.7
                     plt.plot(cumulative_returns_norm.index, cumulative_returns_norm[col],
                              label=label, linewidth=linewidth, alpha=alpha)

                 plt.title("Growth of \$1: Strategy Comparison (2000-Present)")
                 plt.ylabel("Portfolio Value (Log Scale)")
                 plt.xlabel("Date")
                 plt.yscale('log')
                 plt.legend(loc='upper left', fontsize='medium') # Slightly larger legend
                 plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                 plt.tight_layout()
                 if SAVE_FIGS:
                     fpath = os.path.join(FIG_PATH, "output_cumulative_returns.png")
                     plt.savefig(fpath, dpi=FIG_DPI)
                     print(f"Saved: {fpath}")
                 plt.show()
            else:
                 print("Skipping Cumulative Return plot due to no valid data.")


            # Plot 2: Sharpe Ratio and Max Drawdown Bar Chart
            print("Plotting Sharpe Ratio & Max Drawdown...")
            plot_df_metrics = metrics_df[['Sharpe', 'Max Drawdown']].dropna().sort_values('Sharpe', ascending=False) # Sort bars by Sharpe
            if not plot_df_metrics.empty:
                fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted size
                indices = np.arange(len(plot_df_metrics))
                bar_width = 0.35

                rects1 = ax.bar(indices - bar_width/2, plot_df_metrics['Sharpe'], bar_width, label='Sharpe Ratio', color='tab:blue')
                # Create secondary y-axis for Max Drawdown for better scale visibility
                ax2 = ax.twinx()
                rects2 = ax2.bar(indices + bar_width/2, plot_df_metrics['Max Drawdown'], bar_width, label='Max Drawdown (Right Axis)', color='tab:red', alpha=0.7)

                ax.set_xticks(indices)
                ax.set_xticklabels([label.replace('_', ' ') for label in plot_df_metrics.index], rotation=45, ha='right')
                ax.set_ylabel('Sharpe Ratio', color='tab:blue')
                ax.tick_params(axis='y', labelcolor='tab:blue')
                ax2.set_ylabel('Max Drawdown', color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}')) # Format MDD axis as percentage

                ax.set_title('Sharpe Ratio and Maximum Drawdown by Strategy (2000–2024)')
                # Combine legends
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper right')
                ax.grid(axis='y', linestyle='--', linewidth=0.5, color='tab:blue', alpha=0.5)
                ax2.grid(axis='y', linestyle=':', linewidth=0.5, color='tab:red', alpha=0.5)

                # Add value labels
                ax.bar_label(rects1, fmt='{:.2f}', padding=3, fontsize='small')
                ax2.bar_label(rects2, fmt='{:.1%}', padding=3, fontsize='small')

                fig.tight_layout() # Use fig.tight_layout() when using twinx
                if SAVE_FIGS:
                    fpath = os.path.join(FIG_PATH, "output_sharpe_mdd_bars.png")
                    plt.savefig(fpath, dpi=FIG_DPI)
                    print(f"Saved: {fpath}")
                plt.show()
            else:
                print("Skipping Sharpe/MDD plot due to missing data.")

            # --- Add more plots here if needed (e.g., weight plots) ---
            # Example: Plotting weights (similar structure to original code)
            # Consider creating a generic plotting function for weights as well

    else:
        print("Could not load data. Exiting.")

    print("\n--- Script Finished ---")