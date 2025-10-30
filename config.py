import os

# -------------------SEED-------------------
GLOBAL_SEED: int = 42                                   # Global seed for reproducibility

# ------------------GENERAL-------------------
SYMBOL: str = 'SPY'                                     # Stock symbol for which the model will be trained and the features will be calculated
TRADING_DAYS_PER_YEAR: int = 252                        # Number of trading days per year

# ------------------LOGGING------------------
LOG_LEVEL: str = 'DEBUG'                                # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_BACKUPS: int = 180                                  # Logs are stored for 180 days
PATH_LOGS: str = '/logs'                                # Path to store the logs

# ------------------DATABASE------------------
DB_NAME: str = 'historical_option_data.db'              # Name of the database
PATH_TO_OPTIONS_CSV: str = 'data/options_data.csv'      # Path to the CSV file containing the option data
PATH_TO_FEATURES_CSV: str = 'data/features.csv'         # Path to the CSV file containing the features data
OPTIONS_DATA_TABLE: str = 'options_data'                # Name of the table containing the option data in the database
FEATURES_DATA_TABLE: str = 'features'                   # Name of the table containing the features in the database

# ------------------FEATURES------------------
DOWNLOAD_FEATURES: bool = False                         # Download features from Yahoo Finance
SAVE_FEATURES_TO_CSV: bool = False                      # Save features to a CSV file
VOLATILITY_TIMEFRAMES: list[int] = [5, 10, 20, 50]      # List of timeframes (in days) to calculate the volatility features
MOMENTUM_TIMEFRAMES: list[int] = [5, 10, 20, 50]        # List of timeframes (in days) to calculate the momentum features

# ------------------DATA SPLITTING------------------
TRAINING_SET_PERCENTAGE: float = 70                     # Percentage of data used for training
VALIDATION_SET_PERCENTAGE: float = (1 - 0.5/(TRAINING_SET_PERCENTAGE/100)) * 100  # Percentage of data used for validation during training out of the training set. Ensure as 50/20/30 split between train/val/test.
MIN_MONEYNESS: float = 0.8                              # Minimum moneyness of options to include
MAX_MONEYNESS: float = 1.2                              # Maximum moneyness of options to include

# ------------------HYPERPARAMETER TUNING & TRAINING------------------
RUN_HYPERPARAMETER_TUNING: bool = True                  # Run hyperparameter tuning
TUNER_DIR: str = 'hyperband_tuner'                      # Path to store the hyperparameter tuning results      
SETTINGS = {
    "num_epochs_final_training": 5,                   # The number of epochs to train the final model after hyperparameter tuning is complete, or during each retraining fold.
    "early_stopping_patience": 50,                      # Number of epochs with no improvement after which training will be stopped
    "overwrite_tuner": False,                           # If True, deletes the previous tuner directory and starts a fresh hyperparameter search. If False, attempts to resume the last search.
    "initial_guess": [0.04, 2.0, 0.07, 0.5, -0.7],      # Initial guess for the Heston parameters [v0, kappa, theta, sigma, rho]. Used to initialize the neural network's bias to start near a reasonable solution.
    "upper_bounds_sigmoid": [1.0, 5.0, 1.0, 2.0],       # The upper bounds for the first four Heston parameters [v0, kappa, theta, sigma]. A sigmoid activation function ensures the model's output for these parameters stays below these values.
    "max_epochs_tuning": 40,                            # Maximum number of epochs for each trial during hyperparameter tuning
    "executions_per_trial": 1,                          # Number of models to build and fit for each trial to reduce variance
    "hyperband_factor": 4,                              # A parameter for the Hyperband algorithm that controls the reduction rate. At each stage, only the top 1/factor trials are kept.
    "h_relative": 1e-5,                                 # Relative step size for finite difference gradient approximation
    "num_threads": os.cpu_count(),                      # Number of threads to use for parallel processing. Defaults to the number of CPU cores.
    "MIN_FOLD_DAYS": 20,                                # Minimum number of days per fold when performance is monitored and model is recalibrated
    "MAX_FOLD_DAYS": 100,                               # Maximum number of days per fold when performance is monitored and model is recalibrated
    "RECALIBRATION_THRESHOLD_FACTOR": 1.15,             # The factor by which the new error mean must exceed the previous mean to trigger a recalibration. 1.15 means a 15% increase.
    "gradient_method": "forward",                       # Options: "central" (more accurate) or "forward" (faster ~ 1.5x speedup) 
    "OPTION_SUBSAMPLE_PERCENTAGE": 80.0                 # Percentage of options to use per day to speed up training. Set to 100.0 to use all options.
}                     

# ------------------MODELS------------------
MODELS_DIR: str = 'trained_models'                      # Path to store the trained models
# Path to a pretrained model (.h5) file or None (a new model will be trained). Make sure that the setting
# RUN_HYPERPARAMETER_TUNING in the section HYPERPARAMETER TUNING above is set to False when using a pretrained model.
USE_MODEL: str|None = None #'trained_models/SPY_20250922_163422.weights.h5' 

# ------------------RECALIBRATION CHECKS------------------
RECALIBRATION_THRESHOLD_FACTOR: float = 1.2             # The factor by which the new error mean must exceed the previous mean to trigger a performance degradation warning. 1.20 means a 20% increase. This number should be between > 1.0

# ------------------OUTPUT DIRECTORIES------------------
OUTPUT_BASE_DIR: str = 'results'                        # Base directory to store all run outputs