
## About the Project

todo

## 1. Installation
1. Create a new python virtual environment with 'python > 3.10'
2. Install requirements
   ```
   pip install -r requirements.txt
   ```
3. Install setup
   ```
   python setup.py install
   ```

## 2. Usage

Invoke your training script (e.g. `train.py`) with the following options:

```bash
python __main__.py [OPTIONS]
```

### Data & Domain

* `--domain <str>`
  Dataset domain identifier.
  **Default:** `CC__pl_4_5_6`

* `--folder-data <str>`
  Path where raw or intermediate data lives (or will be built).
  **Default:** `out/NN/Training`

* `--dir-save-data <str>`
  Directory into which processed data will be saved.
  **Default:** `data`

* `--unreachable-state-value <int>`
  Numeric value to assign unreachable states in your distance matrix.
  **Default:** `1000000`

* `--test-size <float>`
  Fraction of the dataset to reserve for testing (between 0.0 and 1.0).
  **Default:** `0.2`

### Model & Training

* `--model-name <str>`
  Name to assign to the trained estimator (used for filenames, logging, etc.).
  **Default:** `distance_estimator`

* `--n-train-epochs <int>`
  Number of epochs for training the neural network.
  **Default:** `500`

* `--batch-size <int>`
  Batch size for the training loop.
  **Default:** `2048`

* `--seed <int>`
  Random seed (for reproducibility of splits, weight initialization, etc.).
  **Default:** `42`

### Actions (boolean flags)

Each of these accepts `true` or `false` (case-insensitive).

* `--build-data <bool>`
  Whether to (re)build the processed dataset before training.
  **Default:** `true`

* `--train <bool>`
  Whether to actually train the model after building/loading data.
  **Default:** `true`

### Feature Options

* `--kind-of-ordering <hash|map>`
  Strategy for ordering your state representations.
  **Default:** `hash`

* `--kind-of-data <merged|separated>`
  Whether to merge all data into one set, or keep per-domain splits separate.
  **Default:** `merged`

* `--use-goal <bool>`
  Include goal information as part of the input features.
  **Default:** `false`

* `--use-depth <bool>`
  Include depth (search depth or graph distance) as part of the input features.
  **Default:** `false`

### Logging & Debugging

* `--verbose <bool>`
  Print detailed evaluation errors and training progress.
  **Default:** `false`

---

### Example

Load the pre-trained model:

```bash
python train.py \
  --build-data false \
  --train false \
  --domain CC__pl_4_5_6 \
  --folder-data out/NN/Training \
  --dir-save-data data
```

Build data and train new model:

```bash
python train.py \
  --build-data true \
  --train true \
  --use-goal true \
  --use-depth true \
  --n-train-epochs 300 \
  --batch-size 1024
```

Adjust any of the flags above to customize data handling, model training, and logging.
