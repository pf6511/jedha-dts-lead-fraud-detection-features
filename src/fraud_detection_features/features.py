import numpy as np
import pandas as pd

from .config import (
    TARGET_COLUMN,
    CARD_NUM_COLUMN,
    AMOUNT_COLUMN,
    TIMESTAMP_COLUMN,
    CLIENT_LOCATION_LAT_COLUMN,
    CLIENT_LOCATION_LON_COLUMN,
    MERCHANT_LOCATION_LAT_COLUMN,
    MERCHANT_LOCATION_LON_COLUMN,
    DISTANCE_FEATURE,
    CARD_TRANSACTION_COUNT_FEATURE,
    CARD_FRAUD_COUNT_FEATURE,
    CARD_FRAUD_RATE_FEATURE,
    CARD_AVG_AMOUNT_FEATURE,
)

def add_distance_feature(dtf:pd.DataFrame) -> pd.DataFrame:
    """Add the distance between cardholder and merchant in kilometers."""
    ...
    dtf = dtf.copy()
    lat1 = np.radians(dtf[CLIENT_LOCATION_LAT_COLUMN])
    lon1 = np.radians(dtf[CLIENT_LOCATION_LON_COLUMN])

    lat2 = np.radians(dtf[MERCHANT_LOCATION_LAT_COLUMN])
    lon2 = np.radians(dtf[MERCHANT_LOCATION_LON_COLUMN])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    dtf[DISTANCE_FEATURE] = 2 * 6371 * np.arcsin(np.sqrt(a))
    return dtf


def add_credit_card_history_features(dtf:pd.DataFrame, p0: float,alpha: float = 10.0) -> pd.DataFrame:
    dtf = dtf.copy()
    dtf[CARD_TRANSACTION_COUNT_FEATURE] = (
        dtf.groupby(CARD_NUM_COLUMN).cumcount()
    )

    dtf[CARD_FRAUD_COUNT_FEATURE] = (
        dtf.groupby(CARD_NUM_COLUMN)[TARGET_COLUMN]
            .cumsum()
        - dtf[TARGET_COLUMN]
    )

    dtf[CARD_FRAUD_RATE_FEATURE] = (
        dtf[CARD_FRAUD_COUNT_FEATURE] + alpha * p0
    ) / (
        dtf[CARD_TRANSACTION_COUNT_FEATURE] + alpha
    )
    return dtf


def add_card_average_amount(dtf: pd.DataFrame) -> pd.DataFrame:
    """Add the historical average transaction amount per card."""
    ...
    card_amount_sum_temp_col = "card_amount_sum"
    dtf = dtf.copy()
    dtf[card_amount_sum_temp_col] = (
        dtf.groupby(CARD_NUM_COLUMN)[AMOUNT_COLUMN].cumsum()
        - dtf[AMOUNT_COLUMN]
    )

    dtf[CARD_AVG_AMOUNT_FEATURE] = (
        dtf[card_amount_sum_temp_col]
        / dtf[CARD_TRANSACTION_COUNT_FEATURE]
    )
    return dtf.drop(columns=card_amount_sum_temp_col)


def build_training_features(
    dtf: pd.DataFrame,
    p0: float,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """
    Build features for historical model training.

    Transactions are sorted chronologically so that historical
    features only use transactions preceding the current transaction.
    """
    dtf = (
        dtf.sort_values(by=TIMESTAMP_COLUMN)
        .copy()
    )

    dtf = add_distance_feature(dtf)

    dtf = add_credit_card_history_features(
        dtf,
        p0=p0,
        alpha=alpha,
    )

    return add_card_average_amount(dtf)


def build_inference_features(
    payment: pd.DataFrame,
    history: pd.DataFrame,
    p0: float,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """
    Build features for a single real-time transaction.
    Add features to existing columns

    Parameters
    ----------
    payment:
        Current transaction to predict. Expected to contain one row.

    history:
        Previously consumed transactions for the same card.
        The current transaction must not be included.

    p0:
        Training fraud prevalence used for smoothing.

    alpha:
        Smoothing strength.
    """
    payment = payment.copy()
    history = history.copy()

    transaction_count = len(history)

    fraud_count = (
        history[TARGET_COLUMN].sum()
        if not history.empty
        else 0
    )

    fraud_rate = (
        (fraud_count + alpha * p0)
        / (transaction_count + alpha)
    )

    average_amount = (
        history[AMOUNT_COLUMN].mean()
        if not history.empty
        else np.nan
    )

    payment = add_distance_feature(payment)

    payment[CARD_TRANSACTION_COUNT_FEATURE] = transaction_count
    payment[CARD_FRAUD_COUNT_FEATURE] = fraud_count
    payment[CARD_FRAUD_RATE_FEATURE] = fraud_rate
    payment[CARD_AVG_AMOUNT_FEATURE] = average_amount

    return payment
