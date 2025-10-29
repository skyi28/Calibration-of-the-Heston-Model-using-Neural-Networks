# Adaptive Heston Model Calibration with Neural Networks

This repository contains a comprehensive framework for calibrating the Heston stochastic volatility model to real-world options data. It leverages a deep neural network to learn the complex relationship between market features and the Heston parameters, featuring an adaptive recalibration mechanism to ensure the model stays current with changing market dynamics.

## Overview

Traditional methods for calibrating the Heston model often involve complex optimization problems that can be slow and may converge to local minima. This project presents a machine learning-based approach where a neural network directly predicts the optimal Heston parameters (`v0`, `kappa`, `theta`, `sigma`, `rho`) from a rich set of financial and economic indicators.

The key innovation is the **adaptive window evaluation**. The model is trained on an initial block of historical data and then evaluated on subsequent days. Its performance is continuously monitored, and a statistical change-point detection algorithm automatically triggers a recalibration (retraining) when the model's accuracy degrades, simulating a real-world production environment.

## Key Features

-   **Deep Learning for Calibration**: Utilizes a TensorFlow/Keras neural network with a residual architecture to predict Heston parameters accurately.
-   **Adaptive Recalibration**: Implements a forward-chaining methodology that automatically retrains the model when performance deteriorates, ensuring robustness to market regime shifts.
-   **Advanced Financial Modeling**: Integrates the powerful **QuantLib** library for analytically pricing European options under the Heston model.
-   **Custom Gradient Training**: A custom TensorFlow gradient (`tf.custom_gradient`) allows the neural network to be trained end-to-end, even though the loss function is calculated by the external QuantLib library.
-   **Comprehensive Feature Engineering**: Automatically downloads and processes a wide range of features, including historical volatility, momentum, VIX, SKEW, and the US Treasury yield curve.
-   **Efficient Data Handling**: Uses **DuckDB** for high-performance storage and querying of options and feature data.
-   **Hyperparameter Optimization**: Employs **Keras Tuner** to systematically find the optimal neural network architecture.
-   **Detailed Evaluation & Visualization**: Generates a suite of plots and metrics to analyze the model's performance over time, including parameter drift, daily pricing errors, and implied volatility surfaces.

## How It Works

1.  **Data Ingestion & Feature Engineering**: The script begins by loading raw options data. If specified, it fetches a wide array of features from sources like Yahoo Finance and FRED. These include underlying-specific features (volatility, momentum) and macroeconomic indicators (VIX, SKEW, yield curve). All data is stored in a local DuckDB database.

2.  **Data Partitioning**: The historical data is split chronologically into an initial training set and a subsequent monitoring set.

3.  **Hyperparameter Tuning (Optional)**: Keras Tuner is used on the initial training data to find the best neural network architecture (number of layers, neurons, dropout rate, etc.) and learning rate.

4.  **Initial Model Training**: A neural network is trained on the initial dataset. The model takes the daily market features as input and outputs the five Heston parameters. The loss function is the weighted Mean Squared Error (MSE) between the Heston model's option prices and the actual market prices.

5.  **Adaptive Window Evaluation**:
    *   The trained model is used to predict parameters and price options for each day in the monitoring set.
    *   Daily performance metrics (Mean Absolute Error, Root Mean Squared Error) are recorded.
    *   A change-point detection algorithm (`ruptures` library) analyzes the stream of error metrics.
    *   If a statistically significant increase in error is detected (or a maximum time window is reached), a **recalibration is triggered**.
    *   The model is then retrained (fine-tuned) on an updated dataset that includes the data from the period it just evaluated.
    *   This process repeats until all monitoring data is processed.

6.  **Results & Analysis**: At the end of the run, the script saves all collected data (daily metrics, fold summaries, predicted parameters) to the `results/` directory and generates a series of analytical plots.

## Project Structure
```
.
├── data/                   # Storage for CSV files and the DuckDB database
├── logs/                   # Log files generated during script execution
├── results/                # Output directory for plots, metrics, and model weights
├── config.py               # Main configuration file for the project
├── database.py             # Helper module for database interactions
├── plotting.py             # Contains all functions for generating plots
├── script.py               # The main executable script for running the entire pipeline
├── utils.py                # Utility functions for downloading and processing data
└── README.md               # This file
```

### Prerequisites

-   Python 3.9+
-   An internet connection for downloading data and dependencies.

#### Setting up the Sample Data

To allow for immediate testing and demonstration of the project's capabilities, a subsample of the full dataset has been included in the repository. This data covers the ticker SPY and is located in the data/ directory.

The provided files are:

    historical_option_data_SPY.csv

    features_SPY.csv

Action Required: Before you can run the main script.py, you must rename these files. The project's configuration points to generic filenames, so the specific ticker name must be removed.

Please rename the files as follows:

    Rename historical_option_data_SPY.csv → historical_option_data.csv

    Rename features_SPY.csv → features.csv

You can do this manually or by using the command line.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Heston-Calibration-NN.git
    cd Heston-Calibration-NN
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution

1.  **Configure the run**: Open the `config.py` file to set key parameters:
    *   `SYMBOL`: The underlying stock symbol to analyze (e.g., 'SPY').
    *   `DOWNLOAD_FEATURES`: Set to `True` on the first run to download and process all required market data. You can set it to `False` on subsequent runs to use the cached data.
    *   `RUN_HYPERPARAMETER_TUNING`: Set to `True` to run the Keras Tuner search. This can be time-consuming. Once complete, the best parameters are saved, and you can set this to `False`.
    *   Other parameters related to model training, data splits, and evaluation can also be adjusted.

2.  **Run the main script**:
    ```bash
    python script.py
    ```

The script will log its progress to the console and to a file in the `logs/` directory.

## Results and Evaluation

All outputs from a run are saved in a uniquely timestamped sub-directory inside `results/`. This includes:

-   **CSV Files**:
    -   `daily_metrics.csv`: Day-by-day MAE and RMSE of the model.
    -   `evaluation_summary.csv`: A summary of each adaptive fold (duration, average error, reason for ending).
    -   `model_parameters.csv`: The predicted Heston parameters at the start of each fold.
    -   `option_level_results.csv`: Detailed pricing results for individual options.
-   **Plots**: Visualizations of daily performance, parameter drift over time, and analysis of pricing errors by moneyness and maturity.
-   **Model Weights**: The trained model weights for each fold are saved as `.h5` files, allowing you to resume or analyze a specific model.

## Dependencies

This project relies on a number of powerful open-source libraries:

-   **TensorFlow & Keras Tuner**: For building, training, and optimizing the neural network.
-   **QuantLib**: The premier open-source library for quantitative finance.
-   **DuckDB**: An in-process SQL OLAP database management system.
-   **scikit-learn**: For data preprocessing (StandardScaler).
-   **pandas & NumPy**: For data manipulation and numerical operations.
-   **yfinance**: For downloading stock market data.
-   **ruptures**: For change-point detection in the adaptive evaluation loop.
-   **tqdm**: For progress bars.