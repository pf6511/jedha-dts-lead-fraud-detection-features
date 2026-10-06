from fraud_detection_features.features import build_inference_features
import pandas as pd
import pytest
from pathlib import Path


def test_build_inference_features():
    history = pd.DataFrame(
        [
            {
                "cc_num": "123",
                "trans_date_trans_time": "2020-01-01 10:00:00",
                "amt": 100.0,
                "is_fraud": 0,
                "lat": 40.0,
                "long": -70.0,
                "merch_lat": 40.1,
                "merch_long": -70.1,
            },
            {
                "cc_num": "123",
                "trans_date_trans_time": "2020-01-01 11:00:00",
                "amt": 200.0,
                "is_fraud": 1,
                "lat": 40.0,
                "long": -70.0,
                "merch_lat": 40.2,
                "merch_long": -70.2,
            },
        ]
    )

    payment = pd.DataFrame([{
        "cc_num": "123",
        "trans_date_trans_time": "2020-01-01 12:00:00",
        "amt": 300.0,
        "lat": 40.0,
        "long": -70.0,
        "merch_lat": 40.3,
        "merch_long": -70.3,
    }])

    p0 = 0.004565517489582342
    alpha = 10
    features = build_inference_features(
        payment=payment,
        history=history,
        p0=p0,
        alpha=alpha,
    )

    print(type(features))
    assert features["card_transaction_count"].iloc[0] == 2
    assert features["card_fraud_count"].iloc[0] == 1
    assert features["card_fraud_rate"].iloc[0] == pytest.approx((1 + 10 * p0)
        / (2 + alpha))
    assert features["card_avg_amount"].iloc[0] == pytest.approx(300.0/2)
    
