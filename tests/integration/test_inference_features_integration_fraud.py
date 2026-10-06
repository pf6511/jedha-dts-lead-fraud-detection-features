from pathlib import Path

import pandas as pd
import pytest
import json

from fraud_detection_features.features import build_inference_features


TEST_DATA_PATH = Path("tests/data/transactions_test_with_fraud.csv")


def test_build_inference_features_integration():
    df = pd.read_csv(TEST_DATA_PATH)

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    cc_num = df["cc_num"].iloc[0]

    card_transactions = (
        df[df["cc_num"] == cc_num]
        .sort_values("trans_date_trans_time")
        .reset_index(drop=True)
    )

    assert len(card_transactions) >= 2

    # Dernière transaction = transaction à prédire.
    payment = card_transactions.iloc[[-1]]
    #print('type(payment : ', type(payment))
    #print(len(payment))
    #print(payment.to_json())

    # Toutes les transactions précédentes = historique.
    history = card_transactions.iloc[:-1]

    payment_with_features = build_inference_features(
        payment=payment,
        history=history,
        p0=0.004565517489582342,
        alpha=10,
    )


    assert isinstance(payment_with_features, pd.DataFrame)
    assert len(payment_with_features) == 1

    row = payment_with_features.iloc[0]
    
    payload = {
        "transaction": {
            "amt": row["amt"],
            "lat": row["lat"],
            "long": row["long"],
            "city_pop": row["city_pop"],
            "merch_lat": row["merch_lat"],
            "merch_long": row["merch_long"],
            "merchant": row["merchant"],
            "category": row["category"],
            "gender": row["gender"],
            "state": row["state"],
            "job": row["job"],
        },
        "card_features": {
            "distance_km": row["distance_km"],
            "card_transaction_count": row["card_transaction_count"],
            "card_fraud_count": row["card_fraud_count"],
            "card_fraud_rate": row["card_fraud_rate"],
            "card_avg_amount": row["card_avg_amount"],
        },
    }

    print(json.dumps(payload, indent=2, default=str))



    assert payment_with_features["card_transaction_count"].iloc[0] == len(history)
    
    #-------------------------------------
    