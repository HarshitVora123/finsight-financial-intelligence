"""Train an independent baseline on UCI dataset 350 (CC BY 4.0)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brightpath.data.uci import load_uci_credit_default
from brightpath.models.risk import train_risk_model


if __name__ == "__main__":
    cache = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
    x, y = load_uci_credit_default(cache)
    frame = x.copy()
    frame["default_payment_next_month"] = y
    print(train_risk_model(frame, ROOT / "models" / "uci", feature_columns=list(x.columns),
                           target_column="default_payment_next_month", source="UCI_dataset_350"))
