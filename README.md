# Diamond Price Predictor

A Streamlit frontend for the Random Forest diamond-price model from the supplied notebook.

## Project files

```text
.
├── app.py
├── diamond_rf_model.joblib
├── model_metadata.json
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Run locally

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload every file in this folder, including `.streamlit/config.toml` and `diamond_rf_model.joblib`.
3. Commit and push the files to the `main` branch.
4. Open Streamlit Community Cloud and select **Create app**.
5. Choose the GitHub repository, branch `main`, and entrypoint file `app.py`.
6. Select a Python version compatible with the pinned dependencies, then deploy.

The model file is about 62 MB, so it fits below GitHub's regular 100 MB per-file limit. Do not exclude it using `.gitignore`.

## Important preprocessing used

- Duplicate rows removed.
- Columns renamed: `x`, `y`, `z`, and `depth`.
- Rows with zero physical dimensions removed.
- Width and physical depth restricted to less than 15 mm.
- Ordinal mappings from the notebook are used for cut, color, and clarity.
- Random Forest Regressor with 100 trees and `random_state=42`.
