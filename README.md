# MLflow Tracking Workshop Repo

This repository is a small, self-contained **learning playground** for
software engineers who want to understand how **MLflow Tracking** helps
with experiment traceability, reproducibility, and model packaging.

It is intentionally minimal and opinionated: the goal is to be easy to read
and fork, not to be production-grade.

---

## What You Will Learn

- **Core MLflow Tracking concepts**
  - Experiments, runs, params, metrics, tags, and artifacts.
- **Hands-on tracking with scikit-learn**
  - Training a small binary classifier and logging everything about a run.
- **Artifacts and models**
  - Logging plots, reports, and an MLflow Model artifact.
- **UI vs APIs**
  - Using the MLflow UI to inspect runs.
  - Understanding that the same data is available via Python/REST APIs.

The target audience is a software engineer with little or no ML background.

---

## Repository Structure

```text
demo.py         # Main MLflow Tracking demo script
requirements.txt
README.md

mlruns/         # Created after running demo.py: local artifact/metadata store
mlflow.db       # Created after running demo.py: local SQLite backend store
```

Running the demo once will populate `mlruns/` and `mlflow.db` with
experiments, runs, metrics, tags, artifacts, and a logged model.

---

## Setup

You’ll need **Python 3.10+** installed.

From this folder (the repo root), run:

```powershell
# 1) Create a virtual environment
python -m venv .venv

# 2) Activate it (PowerShell)
\.venv\Scripts\Activate.ps1

# If execution policy blocks activation, you can run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install --upgrade pip
pip install -r .\requirements.txt
```

---

## Running the Demo

### 1. Log a Run

With the virtual environment activated, from this folder run:

```powershell
python .\demo.py
```

This script will:
- Load a small breast cancer dataset from scikit-learn.
- Train a `RandomForestClassifier`.
- Log **params** (e.g. `n_estimators`, `max_depth`).
- Log **metrics** (accuracy, ROC AUC).
- Log **tags** (model type, dataset, owner).
- Generate and log **artifacts**:
  - Confusion matrix plot.
  - ROC curve plot.
  - Feature importance plot.
  - Text classification report.
- Log an **MLflow Model** artifact for the trained model.

You should now see new files/folders:
- `mlflow.db` – SQLite DB for **metadata** (experiments, runs, params, metrics, tags).
- `mlruns/` – Directory holding **artifacts** and run metadata.

---

### 2. Explore in the MLflow UI

Still in the activated virtual environment, start the UI:

```powershell
mlflow ui
```

By default this starts a local server at `http://127.0.0.1:5000`.

In the browser:

1. Open the URL printed in the terminal.
2. Select the experiment named **`demo-mlflow-artifacts`**.
3. You should see at least one run named **`random_forest_breast_cancer`**.
4. Click on a run and explore:
   - **Parameters**: hyperparameters used for this run.
   - **Metrics**: accuracy and ROC AUC.
   - **Tags**: model type, dataset, owner.
   - **Artifacts**:
     - Plots (confusion matrix, ROC curve, feature importances).
     - Text report.
     - Logged model under the `model/` artifact.

This is the core **MLflow Tracking** experience:
- Every run has traceable configuration and results.
- Artifacts are attached to the run for later inspection.

Stop the UI with `Ctrl+C` in the terminal.

---

## Mental Model (for Software Engineers)

If you’re used to Git + CI, you can think of MLflow as:

- **Experiments** → like a project or repo for a particular modelling problem.
- **Runs** → like CI build runs or test runs, each with:
  - Params (configuration).
  - Metrics (results).
  - Tags (free-form metadata: environment, owner, etc.).
  - Artifacts (files: models, plots, reports).

Under the hood MLflow separates:

- A **backend store** for metadata (runs, params, metrics, tags).
- An **artifact store** for larger files (plots, models, datasets, etc.).

The **UI is optional**: it’s just a web app talking to the same backend.
You can also access all metadata programmatically via MLflow’s Python and REST APIs to build your own dashboards or automation.

---

## How to Use This Repo

- **As a quick demo**
  - Clone the repo.
  - Follow the **Setup** and **Running the Demo** sections.
  - Use the UI to inspect params, metrics, tags, and artifacts.

- **As a workshop**
  - Change hyperparameters in `demo.py` and re-run to create multiple runs.
  - Compare runs in the UI by metrics and artifacts.
  - Inspect the logged **MLflow Model** and generated `conda.yaml` to see how MLflow captures the environment.

- **As a starting point**
  - Replace the scikit-learn model with your own model code.
  - Extend logging with additional metrics, tags, and artifacts.
  - Point MLflow at a remote tracking server and shared artifact store.

---

## Typical Questions

- **Do I need to run a server to log runs?**  
  No. In this demo, the script writes directly to a local backend/artifact store. You only run `mlflow ui` when you want to inspect things in the browser.

- **What is the `mlflow.db` file?**  
  A local SQLite database used as the **backend store** for metadata.

- **What is `mlruns/`?**  
  A directory that stores per-experiment, per-run folders with artifacts and additional metadata files.

- **Can I get the data without using the UI?**  
  Yes. The UI is just one client. You can use the MLflow Python/REST APIs to list experiments, search runs, read params/metrics/tags, and download artifacts into your own tools.

- **How does this look in a real team setup?**  
  Typically you run a shared MLflow tracking server backed by your company’s DB and object storage. Training jobs send tracking data to that server, and teams consume it via the UI or APIs.

---

## Where to Go Next

If you want to dig deeper:
- Try changing hyperparameters in `demo.py` and re-running to see multiple runs in the UI.
- Compare runs by metrics in the UI.
- Inspect the logged **MLflow Model** and the generated `conda.yaml` to see how MLflow captures the environment for reproducibility.

---

## Serving the Logged Model (Optional)

The MLflow UI lets you **inspect** and **locate** the model artifact, but it does not
directly expose a prediction endpoint. To serve the model locally:

1. In the UI, open the run page and note the **Run ID** (or copy the model URI
   from the `model/` artifact; it will look like `runs:/<run_id>/model`).
2. In a terminal with the virtual environment activated, run:

   ```powershell
   mlflow models serve -m "runs:/<run_id>/model" -p 5001
   ```

3. Send prediction requests as JSON to the REST endpoint, e.g. with `curl`:

   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     --data '{"data": [[0.1, 0.2, 0.3, ...]]}' \
     http://127.0.0.1:5001/invocations
   ```

In practice, teams often:
- Use `mlflow models serve` for quick local testing.
- Or load the model in their own service with
  `mlflow.sklearn.load_model("runs:/<run_id>/model")` and expose an endpoint via
  FastAPI/Flask/etc.

For more details, see the official MLflow docs (Tracking, Models, Models Serving, and Model Registry sections).
