"""Create the six source-backed exploration notebooks in notebooks/."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "01_data_understanding.ipynb": ("Data understanding and validation", [
        "from pathlib import Path\nimport sys, pandas as pd\nROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))\n",
        "from brightpath.data.synthetic import generate_synthetic_data\nfrom brightpath.data.validation import validate_bundle\nbundle = generate_synthetic_data(500, seed=42)\npd.DataFrame(validate_bundle(bundle))\n",
        "bundle.customers.head()\n",
    ]),
    "02_eda_and_statistics.ipynb": ("EDA and statistical investigation", [
        "from pathlib import Path\nimport sys, pandas as pd\nROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))\n",
        "from brightpath.data.synthetic import generate_synthetic_data\nfrom brightpath.features.engineering import build_customer_features\nbundle = generate_synthetic_data(500, seed=42)\nfeatures = build_customer_features(bundle.customers, bundle.transactions, bundle.repayments)\n",
        "from brightpath.evaluation.eda import create_eda_report\nreport = create_eda_report(features, ROOT / 'reports')\nreport['hypothesis_tests']\n",
    ]),
    "03_feature_engineering.ipynb": ("Feature engineering and point-in-time checks", [
        "from pathlib import Path\nimport sys\nROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))\n",
        "from brightpath.data.synthetic import generate_synthetic_data\nfrom brightpath.features.engineering import build_customer_features, feature_dictionary, MODEL_FEATURES\nbundle = generate_synthetic_data(500, seed=42)\nfeatures = build_customer_features(bundle.customers, bundle.transactions, bundle.repayments)\n",
        "import pandas as pd\npd.DataFrame({'feature': MODEL_FEATURES, 'why_it_matters': [feature_dictionary()[f] for f in MODEL_FEATURES]})\n",
        "features[MODEL_FEATURES].describe().T\n",
    ]),
    "04_risk_modeling.ipynb": ("Risk model training, comparison and calibration", [
        "from pathlib import Path\nimport sys, pandas as pd\nROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))\n",
        "from brightpath.data.synthetic import generate_synthetic_data\nfrom brightpath.features.engineering import build_customer_features\nbundle = generate_synthetic_data(800, seed=42)\nfeatures = build_customer_features(bundle.customers, bundle.transactions, bundle.repayments)\n",
        "from brightpath.models.risk import train_risk_model\nmetrics = train_risk_model(features, ROOT / 'models' / 'notebook', seed=42)\nmetrics['evaluation']\n",
        "pd.DataFrame(metrics['cross_validation']).T\n",
    ]),
    "05_customer_segmentation.ipynb": ("Behavioural segmentation and PCA", [
        "from pathlib import Path\nimport sys\nROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))\n",
        "from brightpath.data.synthetic import generate_synthetic_data\nfrom brightpath.features.engineering import build_customer_features\nbundle = generate_synthetic_data(800, seed=42)\nfeatures = build_customer_features(bundle.customers, bundle.transactions, bundle.repayments)\n",
        "from brightpath.models.segmentation import segment_customers\nsegments = segment_customers(features, ROOT / 'reports', seed=42)\nsegments\n",
    ]),
    "06_experimentation.ipynb": ("Randomized recommendation scenario", [
        "from pathlib import Path\nimport sys\nROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))\n",
        "from brightpath.data.synthetic import generate_synthetic_data\nfrom brightpath.features.engineering import build_customer_features\nbundle = generate_synthetic_data(800, seed=42)\nfeatures = build_customer_features(bundle.customers, bundle.transactions, bundle.repayments)\n",
        "from brightpath.evaluation.experimentation import simulate_experiment\nexperiment = simulate_experiment(features, ROOT / 'reports', seed=42)\nexperiment\n",
    ]),
}


def markdown_cell(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": [text]}


def code_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def create() -> None:
    destination = ROOT / "notebooks"
    destination.mkdir(exist_ok=True)
    for filename, (title, code_cells) in SPECS.items():
        cells = [markdown_cell(f"# {title}\n\nReproducible synthetic demonstration; run from the project root. For end-to-end artifacts, use `python scripts/run_pipeline.py`.", "intro")]
        cells.extend(code_cell(source, f"cell-{index}") for index, source in enumerate(code_cells, 1))
        notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                    "language_info": {"name": "python", "version": "3.11+"}},
                    "nbformat": 4, "nbformat_minor": 5}
        (destination / filename).write_text(json.dumps(notebook, indent=2), encoding="utf-8")


if __name__ == "__main__":
    create()
