
## About the Project
This project develops an estimator that predicts how far a given state is from the goal within dynamic epistemic search
trees. Each search space is modeled as a Kripke structure.

## 1. Installation
1. Create a new python virtual environment (with 'python > 3.10'):
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Install setup:
   ```
   python setup.py install
   ```
4. Enable pre‑commit hooks
   ```
   pre-commit install
   ```
5. Run all hooks against existing files
   ```
   pre-commit run --all-files
   ```

## 2. Usage

Launch `__main__.py` and customize your process with the available argparse options:

```bash
python __main__.py [OPTIONS]
```

### Data Options

* `--subset-train <str>...`
  Name(s) of problem subsets to use for training.
  **Default:** `[]`
* `--folder-raw-data <str>`
  Path where raw or intermediate data lives (or will be built).
  **Default:** `out/NN/Training`
* `--dir-save-data <str>`
  Directory into which processed data will be saved.
  **Default:** `data`
* `--unreachable-state-value <int>`
  Numeric value to assign unreachable states in your distance matrix.
  **Default:** `10^6`
* `--test-size <float>`
  Fraction of the dataset to reserve for testing (between 0.0 and 1.0).
  **Default:** `0.2`

### Model & Output Options

* `--folder-model-name <str>`
  Name of the model folder where checkpoints will be stored.
  **Default:** `models`
* `--model-name <str>`
  Name for the distance estimator model (used for filenames and logging).
  **Default:** `distance_estimator`
* `--normalization-constants-name <str>`
  Name of the normalization constants file for rescaling outputs.
  **Default:** `C`
* `--dir-save-model <str>`
  Directory into which trained models will be saved.
  **Default:** `models`
* `--experiment-name <str>`
  Identifier for this experiment's data and model outputs.
  **Default:** `new_exp`

### Training Parameters

* `--n-train-epochs <int>`
  Number of epochs for training the neural network.
  **Default:** `500`
* `--batch-size <int>`
  Batch size for the training loop.
  **Default:** `2048`
* `--seed <int>`
  Random seed for reproducibility of splits and initialization.
  **Default:** `42`

### Boolean Flags

Each of these accepts `true` or `false` (case-insensitive):

* `--build-data <bool>`
  Whether to (re)build the processed dataset before training.
  **Default:** `true`
* `--train <bool>`
  Whether to actually train the model after data is available.
  **Default:** `true`
* `--if-try-example <bool>`
  Run inference on example samples using PyTorch and ONNX.
  **Default:** `false`
* `--use-goal <bool>`
  Include goal information as part of the input features.
  **Default:** `false`
* `--use-depth <bool>`
  Include depth information (e.g., search depth) as a feature.
  **Default:** `false`
* `--verbose <bool>`
  Print detailed evaluation errors and progress logs.
  **Default:** `false`

### Feature Options

* `--kind-of-ordering <hash|map>`
  Strategy for ordering your state representations.
  **Default:** `hash`
* `--kind-of-data <merged|separated>`
  Whether to merge all data into a single set or keep splits separate.
  **Default:** `merged`

---

### Example Commands

Build data and train with goal and depth features enabled:

```bash
python __main__.py \
  --build-data true \
  --train true \
  --use-goal true \
  --use-depth true \
  --n-train-epochs 300 \
  --batch-size 1024
```
