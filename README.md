# Pain Intensity Classification from Skeletal Motion

This project was developed for the first Artificial Neural Networks and Deep Learning (AN2DL) challenge in November 2025. The task was to predict three levels of pain — **no pain, low pain, and high pain** — from sequences of human joint measurements.

Each sample contains 160 timesteps and initially 31 joint features. The dataset is quite imbalanced, with far more no-pain sequences than high-pain ones. We also found that some joint channels were noisy or barely changed over time, so choosing useful features and handling the class imbalance were important parts of the work.

## What we tried

The main approach in the available notebooks is a **BiLSTM with multi-head self-attention**, implemented in TensorFlow/Keras. The pipeline:

- Sorts measurements by sample and time, removes the constant `joint_30` feature, and reshapes the data into sequences of shape `(160, 30)`.
- Uses a stratified 80/20 train/validation split and random oversampling of minority classes in the training set.
- Passes each sequence through two bidirectional LSTM layers, followed by four-head self-attention, temporal average pooling, and a dense classifier with Swish activations.
- Compares six learning-rate and batch-size combinations, selecting the model using validation macro F1.
- Retrains the selected configuration on all labeled data and produces a CSV submission for the challenge.

The [project report](report/AN2DL_First_Challenge_Report.pdf) also describes a second approach combining **1D CNNs, BiLSTMs, attention, and a separate branch for survey features**. It was designed to use both local motion patterns and longer-term context.

## Results

The saved output of the provided BiLSTM notebook reports the following results on its stratified validation split:

| Metric | Score |
| --- | ---: |
| Macro F1 | **0.8553** |
| Weighted F1 | 0.9409 |
| Accuracy | 0.9398 |

The best configuration used a learning rate of `1e-3`, batch size `32`, and oversampling. The high-pain class remained the hardest to classify.

The original report gives approximate validation F1 values of **0.94** for the BiLSTM-attention model and **0.95** for the CNN-BiLSTM approach. These are figures from the report, not values reproduced by the uploaded notebooks; the saved notebook output distinguishes macro F1 from weighted F1 and accuracy.

## Files

- [`notebooks/AN2DL_Challenge1_Code1.ipynb`](notebooks/AN2DL_Challenge1_Code1.ipynb) — BiLSTM, multi-head attention, hyperparameter experiments, and submission code.
- [`notebooks/AN2DL_Challenge1_Code2.ipynb`](notebooks/AN2DL_Challenge1_Code2.ipynb) — second uploaded notebook.
- [`report/AN2DL_First_Challenge_Report.pdf`](report/AN2DL_First_Challenge_Report.pdf) — full challenge report, including both model approaches.

**Note:** The two supplied notebook files are identical. Both contain the BiLSTM-attention implementation; the CNN-BiLSTM model described in the report is not present in these files.

## Running the code

The notebooks were run with TensorFlow/Keras (the saved output shows TensorFlow 2.19.0). Install the dependencies with:

```bash
pip install -r requirements.txt
```

You will need the challenge CSV files, which are **not included** here:

- `pirate_pain_train.csv`
- `pirate_pain_train_labels.csv`
- `pirate_pain_test.csv`

Put these in the notebook runtime's working directory (for example, `/content` in Google Colab) or update the `pd.read_csv(...)` paths. Run the notebook cells in order to train the model, save its `.keras` checkpoint, and generate `submission.csv`.

This is a course challenge project, not a clinically validated pain-assessment system.
