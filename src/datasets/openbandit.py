import os
import pickle
import numpy as np
from obp.dataset import OpenBanditDataset
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_dataset(behavior_policy="random", campaign="all", data_dir="data/datasets"):
    """
    Load the Open Bandit Dataset and convert to standard (X, y) format.

    The dataset is cached after first load to avoid repeated downloading and processing.

    Parameters
    ----------
    behavior_policy : str, default="random"
        Which behavior policy data to load. Options: "random" or "bts"
    campaign : str, default="all"
        Which campaign to load. Options: "all", "men", or "women"
    data_dir : str, default="data/datasets"
        Directory to store cached data

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Context features
    y : ndarray, shape (n_samples,)
        Action labels taken by the behavior policy
    """
    # Generate cache filename based on parameters
    cached_data = f"{data_dir}/obd_{behavior_policy}_{campaign}.pkl"

    # Try to load from cache
    if os.path.exists(cached_data):
        with open(cached_data, 'rb') as f:
            X, y = pickle.load(f)
        return X, y

    # Load the Open Bandit Dataset (data_path=None makes OBP download to default location)
    dataset = OpenBanditDataset(
        behavior_policy=behavior_policy,
        campaign=campaign,
        data_path=None
    )

    # Get the bandit feedback data
    bandit_feedback = dataset.obtain_batch_bandit_feedback()

    # Extract contexts and actions
    X = bandit_feedback['context']  # (n_samples, n_features)
    y = bandit_feedback['action']   # (n_samples,)

    # Cache the processed data
    os.makedirs(data_dir, exist_ok=True)
    with open(cached_data, 'wb') as f:
        pickle.dump((X, y), f)

    return X, y
