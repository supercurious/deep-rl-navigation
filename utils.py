import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

# Helper functions for loading, processing, and plotting

def load_pkl(filepath, verbose=1):
    """Load pkl log file"""
    
    with open(filepath, 'rb') as f:
        logs_loaded = pickle.load(f)
    if verbose:
        print("Loaded: {}".format(filepath))
    
    return logs_loaded

def process_logs(logs, window_size=1):
    """Convert dict to dataframe and simplify run names on columns"""
    
    logs_df = pd.DataFrame.from_dict(logs)
    run_cols = ['run1', 'run2', 'run3', 'run4', 
                'run5', 'run6', 'run7', 'run8'] # rename
    logs_df.columns = run_cols

    # Take moving average
    if window_size > 1:
        for col in logs_df.columns:
            logs_df[col] = logs_df[col].rolling(window_size).mean()

    return logs_df

def df_rolling_mean(df, window=1):
    """Take rolling mean for each col in dataframe"""
    df_out = pd.DataFrame()
    if window > 1:
        for col in df.columns:
            df_out[col] = df[col].rolling(window).mean()
    else:
        df_out = df
    return df_out

def plot_learning(ax, data, alpha=0.5, estimator='median'):
    """Make transient plots"""
    ts = data.index
    if estimator == 'mean':
        est = data.mean(axis=1)
    elif estimator == 'median':
        est = data.quantile(q=0.5, axis=1)
    ci = [data.quantile(q=0.25, axis=1), data.quantile(q=0.75, axis=1)]
    ax.plot(ts, est)
    ax.fill_between(x=ts, y1=ci[0], y2=ci[1], alpha=alpha)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Score')
    ax.grid(1)
    
    return ax

def plot_side_by_side(results, config_labels, configs, alpha=0.5, window=10, 
                      figsize=(10,4), estimator='median'):
    """Side by side comparison plots
    # Parameters
        results (list of dataframes): each dataframe corresponds to results 
            for all random seeds for a particular agent configuration
        config_labels (list of strs): config names
        configs (list): config numbers to plot
        alpha (float): shaded region transparency
        window (int): size of rolling window for moving average
        figsize (tuple): figure size
        estimator (str): 'mean' or 'median'
    """
    
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=figsize)
    ax1 = plot_learning(ax1, df_rolling_mean(results[configs[0]], 
                        window=window), alpha=alpha, estimator=estimator)
    ax1.set_title(config_labels[configs[0]])
    ax2 = plot_learning(ax2, df_rolling_mean(results[configs[1]], 
                        window=window), alpha=alpha, estimator=estimator)
    ax2.set_title(config_labels[configs[1]])

def plot_overlaid(results, config_labels, configs, alpha=0.5, window=10, 
                  figsize=(10,6), estimator='median'):
    """Overlaid comparison plots
    # Parameters
        results (list of dataframes): each dataframe corresponds to results 
            for all random seeds for a particular agent configuration
        configs (list): config numbers to plot
        config_labels (list of strs): config names
        alpha (float): shaded region transparency
        window (int): size of rolling window for moving average
        figsize (tuple): figure size
        estimator (str): 'mean' or 'median'
    """
        
    fig, ax = plt.subplots(figsize=figsize)
    legend_list = []
    for i in configs: 
        ax = plot_learning(ax, df_rolling_mean(results[i], window=window),
                           alpha=alpha, estimator=estimator)
        legend_list.append(config_labels[i])
    ax.legend(legend_list)
    
    return ax