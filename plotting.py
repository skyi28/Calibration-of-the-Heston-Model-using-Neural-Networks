import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def plot_daily_performance(summary_df, daily_metrics_df, output_dir):
    """
    Plot the daily performance metrics and the retraining events.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame containing the retraining events.
    daily_metrics_df : pd.DataFrame
        DataFrame containing the daily performance metrics.
    output_dir : str
        Directory where to save the plot.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    daily_metrics_df['date'] = pd.to_datetime(daily_metrics_df['date'])
    daily_metrics_df = daily_metrics_df.sort_values('date')
    
    ax.plot(daily_metrics_df['date'], daily_metrics_df['rmse'], label='Daily RMSE', color='b', alpha=0.7)
    ax.plot(daily_metrics_df['date'], daily_metrics_df['mae'], label='Daily MAE', color='c', alpha=0.6)
    
    for _, row in summary_df.iterrows():
        end_date = pd.to_datetime(row['end_date'])
        reason = row['end_reason']
        color = 'r' if 'Degradation' in reason else 'g'
        ax.axvline(x=end_date, color=color, linestyle='--', linewidth=1.2, label=f"Retrain ({reason}) on {end_date.date()}")

    ax.set_title('Daily Model Performance and Retraining Events')
    ax.set_xlabel('Date')
    ax.set_ylabel('Error (in $)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'daily_performance_plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved daily performance plot to {save_path}")

def plot_fold_summary(summary_df, output_dir):
    """
    Plot the fold summary: average error vs. fold duration.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame containing the fold summary data.
    output_dir : str
        Directory where to save the plot.

    Returns
    -------
    None
    """
    fig, ax1 = plt.subplots(figsize=(15, 7))
    index = np.arange(len(summary_df))
    bar_width = 0.35
    
    ax1.bar(index - bar_width/2, summary_df['avg_rmse'], bar_width, label='Avg RMSE', color='b')
    ax1.bar(index + bar_width/2, summary_df['avg_mae'], bar_width, label='Avg MAE', color='c')
    ax1.set_xlabel('Fold Number')
    ax1.set_ylabel('Average Error (in $)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(index)
    ax1.set_xticklabels(summary_df['fold_num'])
    
    ax2 = ax1.twinx()
    ax2.plot(index, summary_df['duration_days'], color='r', marker='o', linestyle='--', label='Fold Duration (days)')
    ax2.set_ylabel('Duration (days)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    fig.suptitle('Fold Performance Summary: Error vs. Duration')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, 'fold_summary_plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved fold summary plot to {save_path}")

def plot_parameter_drift(parameter_df, output_dir):
    """
    Plot the parameter drift across folds.

    Parameters
    ----------
    parameter_df : pd.DataFrame
        DataFrame containing the Heston model parameters for each fold.
    output_dir : str
        Directory where to save the plot.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(len(parameter_df.columns)-1, 1, figsize=(15, 12), sharex=True)
    params = [p for p in parameter_df.columns if p != 'fold_num']
    
    for i, param in enumerate(params):
        axes[i].plot(parameter_df['fold_num'], parameter_df[param], marker='o', linestyle='-')
        axes[i].set_ylabel(param)
        axes[i].grid(True)
    
    axes[-1].set_xlabel('Fold Number')
    fig.suptitle('Heston Parameter Drift Across Folds')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(output_dir, 'parameter_drift_plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved parameter drift plot to {save_path}")

def _create_binned_heatmap(pivoted_data, title, colorbar_label, output_filename):
    """
    Create a heatmap plot from pivoted data.

    Parameters
    ----------
    pivoted_data : pd.DataFrame
        DataFrame containing the pivoted data to plot.
    title : str
        Title of the plot.
    colorbar_label : str
        Label of the colorbar.
    output_filename : str
        File path where to save the plot.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(pivoted_data, cmap='YlOrRd') 
    plt.colorbar(label=colorbar_label)
    plt.xticks(np.arange(len(pivoted_data.columns)), pivoted_data.columns, rotation=45)
    plt.yticks(np.arange(len(pivoted_data.index)), pivoted_data.index)
    plt.xlabel('Moneyness')
    plt.ylabel('Time to Maturity')
    plt.title(title)
    
    for i in range(len(pivoted_data.index)):
        for j in range(len(pivoted_data.columns)):
            value = pivoted_data.iloc[i, j]
            if not pd.isna(value):
                plt.text(j, i, f"{value:.3f}", ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved heatmap to {output_filename}")

def plot_binned_rmse_heatmap(all_option_results_df, output_dir):
    """
    Plot a heatmap of the Model RMSE by Moneyness and Maturity.

    Parameters
    ----------
    all_option_results_df : pd.DataFrame
        DataFrame containing the results of the Heston model for all options.
    output_dir : str
        Directory where to save the plot.

    Returns
    -------
    None
    """
    df = all_option_results_df.copy()
    df['sq_error'] = (df['model_price'] - df['market_price'])**2
    
    binned_data = df.groupby(['maturity_bin', 'moneyness_bin'], observed=False).apply(lambda x: np.sqrt(x['sq_error'].mean())).rename('rmse')
    pivoted_data = binned_data.unstack()
    
    _create_binned_heatmap(pivoted_data, 'Heatmap of Model RMSE by Moneyness and Maturity', 'RMSE (in $)',
                           os.path.join(output_dir, 'binned_rmse_heatmap.png'))

def plot_iv_rmse_heatmap(all_option_results_df, output_dir):
    """
    Plot a heatmap of the IV RMSE by Moneyness and Maturity.

    Parameters
    ----------
    all_option_results_df : pd.DataFrame
        DataFrame containing the results of the Heston model for all options.
    output_dir : str
        Directory where to save the plot.

    Returns
    -------
    None
    """
    df = all_option_results_df.copy()
    df['iv_sq_error_bps'] = ((df['model_iv'] - df['market_iv']) * 10000)**2

    binned_data = df.groupby(['maturity_bin', 'moneyness_bin'], observed=False).apply(lambda x: np.sqrt(x['iv_sq_error_bps'].mean())).rename('iv_rmse_bps')
    pivoted_data = binned_data.unstack()

    _create_binned_heatmap(pivoted_data, 'Heatmap of IV RMSE by Moneyness and Maturity', 'IV RMSE (Basis Points)',
                           os.path.join(output_dir, 'binned_iv_rmse_heatmap.png'))

def plot_mape_heatmap(all_option_results_df, output_dir):
    """
    Plot a heatmap of the Mean Absolute Percentage Error (MAPE) by Moneyness and Maturity for options with a market price above $0.10.

    Parameters
    ----------
    all_option_results_df : pd.DataFrame
        DataFrame containing the results of the Heston model for all options.
    output_dir : str
        Directory where to save the plot.

    Returns
    -------
    None
    """
    df = all_option_results_df.copy()
    df = df[df['market_price'] > 0.10]
    df['mape'] = np.abs((df['model_price'] - df['market_price']) / df['market_price']) * 100
    
    binned_data = df.groupby(['maturity_bin', 'moneyness_bin'], observed=False)['mape'].mean()
    pivoted_data = binned_data.unstack()
    
    _create_binned_heatmap(pivoted_data, 'Heatmap of MAPE by Moneyness and Maturity (Options > $0.10)', 'MAPE (%)',
                           os.path.join(output_dir, 'binned_mape_heatmap.png'))

def plot_3d_vol_surface(surface_df, output_dir, date_str):
    """
    Plot a 3D scatter plot of the implied volatility surface for a given date.

    Parameters
    ----------
    surface_df : pd.DataFrame
        DataFrame containing the implied volatility surface data.
    output_dir : str
        Directory where to save the plot.
    date_str : str
        Date string for the plot title.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(surface_df['time_to_maturity'], surface_df['moneyness'], surface_df['market_iv'], 
               c='blue', marker='o', label='Market IV', alpha=0.6)
    ax.plot_trisurf(surface_df['time_to_maturity'], surface_df['moneyness'], surface_df['model_iv'], 
                    cmap='viridis', alpha=0.7, label='Model IV')

    ax.set_title(f'Implied Volatility Surface for {date_str}')
    ax.set_xlabel('Time to Maturity (Years)')
    ax.set_ylabel('Moneyness (Strike / Spot)')
    ax.set_zlabel('Implied Volatility')
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'volatility_surface_{date_str}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 3D volatility surface plot to {save_path}")