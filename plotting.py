import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from typing import List
import seaborn as sns

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
    
def generate_and_save_correlation_heatmap(data_split: List, title: str, output_path: str, feature_cols: List[str]|None = None):
    """
    Calculates the correlation matrix for a given data split and saves it as a heatmap.

    Parameters
    ----------
    data_split : List
        A list of data tuples, e.g., (date, options_df, features_df).
    title : str
        The title for the plot.
    output_path : str
        The full path (including filename) where the plot image will be saved.
    """
    if not data_split:
        print(f"Skipping heatmap generation for '{title}' as data is empty.")
        return

    # Combine all feature DataFrames from the list into a single DataFrame
    # This correctly handles the different tuple formats you have
    if len(data_split[0]) == 3: # (date, options, features)
        features_df = pd.concat([f for _, _, f in data_split], ignore_index=True)
    elif len(data_split[0]) == 4: # (date, options, features, ql_objects)
        features_df = pd.concat([f for _, _, f, _ in data_split], ignore_index=True)
    else:
        print(f"Warning: Unknown data split format for heatmap generation. Skipping '{title}'.")
        return
    
    if feature_cols:
        # Ensure all requested columns exist in the DataFrame
        final_cols = [col for col in feature_cols if col in features_df.columns]
        plot_df = features_df[final_cols]
    else:
        # If no list is provided, fall back to the old behavior (all numeric columns)
        plot_df = features_df.select_dtypes(include=np.number)
    
    # Calculate the correlation matrix
    corr_matrix = plot_df.corr()

    # Generate and save the heatmap
    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f") # annot=False is better for many features
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)  # x-axis feature labels
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() # Close the plot to free up memory
    print(f"Saved correlation heatmap to: {output_path}")

def plot_shap_summary(shap_values: List[np.ndarray], features: pd.DataFrame, output_dir: str, fold_num: int):
    """
    Generates and saves a SHAP beeswarm summary plot for each model output.

    Parameters
    ----------
    shap_values : List[np.ndarray]
        A list of SHAP value arrays, one for each model output.
    features : pd.DataFrame
        The feature data corresponding to the SHAP values.
    output_dir : str
        The directory where the plots will be saved.
    fold_num : int
        The current fold number, used for titling the plot.
    """
    import shap
    param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    
    for i, param_name in enumerate(param_names):
        plt.figure()
        title = f'SHAP Summary for {param_name} (Fold {fold_num})'
        shap.summary_plot(shap_values[i], features, show=False, plot_type="dot", cmap=plt.get_cmap("coolwarm"))
        plt.title(title)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'shap_summary_{param_name}_fold_{fold_num}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved SHAP summary plot to {save_path}")

def plot_shap_feature_importance(shap_values: List[np.ndarray], features: pd.DataFrame, output_dir: str, fold_num: int):
    """
    Generates and saves a SHAP feature importance bar plot for each model output.

    Parameters
    ----------
    shap_values : List[np.ndarray]
        A list of SHAP value arrays, one for each model output.
    features : pd.DataFrame
        The feature data corresponding to the SHAP values.
    output_dir : str
        The directory where the plots will be saved.
    fold_num : int
        The current fold number, used for titling the plot.
    """
    import shap
    param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    
    # Create a colormap instance
    cmap = plt.get_cmap("coolwarm")
    
    for i, param_name in enumerate(param_names):
        plt.figure()
        title = f'SHAP Feature Importance for {param_name} (Fold {fold_num})'
        
        # To use a colormap with the bar plot, we need to pass an array of colors.
        # We can create a simple gradient for visualization purposes.
        feature_names = features.columns
        num_features = len(feature_names)
        colors = cmap(np.linspace(0, 1, num_features))
        
        shap.summary_plot(shap_values[i], features, show=False, plot_type="bar", color=colors)
        plt.title(title)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'shap_importance_{param_name}_fold_{fold_num}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved SHAP feature importance plot to {save_path}")
        
def save_statistical_tables(df: pd.DataFrame, output_dir: str):
    """
    Calculates and saves descriptive statistics for a DataFrame to text files.

    This function generates three files:
    1. A summary of main descriptive statistics (count, mean, std, etc.).
    2. A summary of skewness and kurtosis.
    3. A combined table containing all of the above metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.
    output_dir : str
        The directory path where the summary text files will be saved.
    """
    print(f"Generating statistical tables in directory: {output_dir}")
    
    # --- Action 1: Save the main descriptive statistics summary ---
    try:
        summary = df.describe()
        summary_path = os.path.join(output_dir, "descriptive_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Main Descriptive Statistics Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(summary.to_string())
        print(f"Saved descriptive summary to {summary_path}")
    except Exception as e:
        print(f"Error saving descriptive summary: {e}")

    # --- Action 2: Save skewness and kurtosis summary ---
    try:
        skew = df.skew(numeric_only=True)
        kurt = df.kurtosis(numeric_only=True)
        skew_kurt_df = pd.DataFrame({'skewness': skew, 'kurtosis': kurt})
        
        skew_kurt_path = os.path.join(output_dir, "skew_kurtosis_summary.txt")
        with open(skew_kurt_path, "w") as f:
            f.write("Skewness and Kurtosis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(skew_kurt_df.to_string())
        print(f"Saved skew/kurtosis summary to {skew_kurt_path}")
    except Exception as e:
        print(f"Error saving skew/kurtosis summary: {e}")
        
    # --- Action 3: Save a combined table ---
    try:
        # To combine, we append the skew/kurtosis DataFrame to the describe() output
        combined_stats = pd.concat([summary, skew_kurt_df.T])
        combined_path = os.path.join(output_dir, "all_statistics.txt")
        with open(combined_path, "w") as f:
            f.write("Combined Statistical Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(combined_stats.to_string())
        print(f"Saved all statistics to {combined_path}")
    except Exception as e:
        print(f"Error saving combined statistics: {e}")


def generate_statistical_plots(df: pd.DataFrame, output_dir: str):
    """
    Generates and saves a grid of histograms and box plots for all numerical columns.

    For readability, this function automatically splits the plots into multiple
    image files if the number of numerical features is large. Each plot file
    contains a grid with histograms in the top row and corresponding box plots
    in the bottom row.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.
    output_dir : str
        The directory path where the plot images will be saved.
    """
    print(f"Generating statistical plots in directory: {output_dir}")
    
    # Identify all numerical columns for plotting
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numerical_cols:
        print("No numerical columns found to generate plots.")
        return

    # --- Architectural Decision ---
    # To prevent creating overly wide and unreadable plots, we limit the number
    # of feature columns per image file. This is more robust than trying to fit
    # dozens of plots into a single figure.
    COLS_PER_PLOT = 6
    col_chunks = [numerical_cols[i:i + COLS_PER_PLOT] for i in range(0, len(numerical_cols), COLS_PER_PLOT)]

    for i, chunk in enumerate(col_chunks):
        num_cols_in_chunk = len(chunk)
        
        # Create a subplot grid with 2 rows (histograms, box plots) and N columns
        fig, axes = plt.subplots(2, num_cols_in_chunk, figsize=(num_cols_in_chunk * 5, 10))
        
        # Ensure 'axes' is always a 2D array for consistent indexing, even with one column
        if num_cols_in_chunk == 1:
            axes = np.array(axes).reshape(2, 1)

        for j, col_name in enumerate(chunk):
            # Top Row: Histograms
            hist_ax = axes[0, j]
            sns.histplot(data=df, x=col_name, ax=hist_ax, kde=True)
            hist_ax.set_title(f"Histogram of {col_name}")
            hist_ax.set_xlabel("")  # Clean up x-label for clarity

            # Bottom Row: Box Plots
            box_ax = axes[1, j]
            sns.boxplot(data=df, y=col_name, ax=box_ax)
            box_ax.set_title(f"Box Plot of {col_name}")
            box_ax.set_ylabel(col_name) # Keep y-label as it's informative for box plots

        plt.tight_layout()

        # Save the figure to a file
        plot_filename = f"combined_statistics_plot_part_{i+1}.png"
        save_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(save_path)
            print(f"Saved plot chunk {i+1} to {save_path}")
        except Exception as e:
            print(f"Error saving plot {save_path}: {e}")
        finally:
            # Ensure the figure is closed to free up memory
            plt.close(fig)