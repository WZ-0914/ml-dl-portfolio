# Implementation and evidence review

Review date: October 29, 2026. This review compares the downloaded Kaggle notebook with the supplied course notebook. It does not execute training or alter the archived code. The later implementation in `src/pipeline.py` addresses the model and workflow issues below; its full runtime checks remain pending. Cell numbers below are one-based positions in the downloaded notebook, including Markdown cells.

## Findings

### 1. A Markdown paragraph was saved as executable code

Cell 2 contains dataset setup instructions. Its type is `code`, whereas the same cell in the template is `markdown`. The saved output contains `SyntaxError: invalid syntax`; later cells have no execution counts or outputs.

The original export is preserved unchanged. The new script contains executable code only and does not use this cell.

### 2. The CNN's flattened feature dimension does not match the input

The preprocessing pipeline produces RGB tensors of size 128 × 128. Cell 22 uses two unpadded 5 × 5 convolutions, each followed by 2 × 2 max pooling with stride 2:

| Stage | Channels | Height × width |
| --- | ---: | --- |
| Input | 3 | 128 × 128 |
| First convolution | 6 | 124 × 124 |
| First pooling | 6 | 62 × 62 |
| Second convolution | 16 | 58 × 58 |
| Second pooling | 16 | 29 × 29 |

Each example therefore has 13,456 features. The notebook uses `x.view(-1, 16*5*5)` and a 400-input linear layer. For the configured training batch of 64, the reshape cannot succeed. For some other batch sizes, this reshape could instead corrupt the batch dimension.

The new script uses `nn.Flatten()` and `nn.Linear(16 * 29 * 29, 120)`. This is a new implementation revision, not a recovered historical result.

### 3. The classifier, loss, and prediction rule disagree

Cell 22 emits one sigmoid probability per example. Cell 25 configures `CrossEntropyLoss`, and validation/prediction choose the maximum along the class dimension. Cross-entropy with integer labels 0 and 1 requires two class logits for this workflow. An argmax across a one-element dimension always selects index 0.

The new script outputs two raw logits, uses cross-entropy for training, and obtains predictions from the two-class output. This fixes the intended classification contract; runtime validation is still pending.

### 4. Validation still records gradients

The inherited validation loop calls `model.eval()` but does not disable autograd. Evaluation mode changes model behavior; it does not disable gradient recording.

The new script disables gradient recording for validation and defaults to batches of 64 for training, validation, and prediction.

### 5. The archive is not a complete reproducible environment

The notebook uses Kaggle-specific paths. The provided environment file leaves PyTorch and torchvision unpinned and does not explicitly list several imported packages, including pandas, matplotlib, seaborn, and Keras. Notebook metadata reports Python 3.7.12; the environment file specifies Python 3.9. These records do not establish the exact environment used for any successful run.

The inherited checkpoint code saves complete model objects. The new script saves model-weight dictionaries and metadata, records framework versions, and predicts from ordinary image folders. The original competition file was not deserialized during this review.

## What was verified

- All 31 cells in the downloaded notebook were compared with the starter notebook.
- Python syntax was checked statically for every code cell; the dataset instruction cell is the only syntax failure found. Syntax checks do not establish runtime correctness.
- Training CSV: 50,000 rows, 25,000 examples per label.
- Validation CSV: 2,000 rows, 1,000 examples per label.
- No duplicate image IDs within either supplied split; no shared image IDs between them.
- Sample submission: 2,000 rows, with `id` and `label` columns.
- The downloaded notebook has one saved error output and no saved metric outputs.

No dataset images were inspected, no training or inference was run, and no Kaggle score was independently verified. The author recalls a score of approximately 92%; this is reported in the project summary as recollection, not a recovered measurement. The large test file was not needed for these checks.

## Optional evidence for the historical record

Additional evidence would strengthen the historical record:

1. A successful Kaggle notebook execution or link, if available.
2. Exact public/private score and ranking from a submission record, to supplement the recalled approximate 92% score.
3. Any experiment notes explaining which models or hyperparameters were tried and what was learned.

Saved checkpoints and a generated submission file are useful optional evidence. The full CelebA image collection is needed only for rerunning training or additional image-based analysis, not for preparing this portfolio entry.
