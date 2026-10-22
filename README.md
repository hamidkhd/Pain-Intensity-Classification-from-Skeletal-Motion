# Pain Intensity Classification from Skeletal Motion

This repository contains our work for the first Artificial Neural Networks and Deep Learning (AN2DL) challenge, completed in November 2025. The task was to classify human movement sequences into three pain levels: **no pain, low pain, and high pain**.

Each sample contains 160 timesteps of skeletal joint measurements. The main challenge was the class imbalance: most sequences belonged to the no-pain class, while high-pain examples were rare. Some joint features were also noisy or almost constant, making it harder to learn useful motion patterns.

## Approach

### Preprocessing

We first organized the joint measurements into fixed-length sequences. The available notebooks include:

- Sorting the data by sample and timestep, then reshaping it into sequences of 160 frames.
- Removing `joint_30`, which was nearly constant, leaving 30 joint features per frame.
- Splitting the labeled data into training and validation sets (80/20), with stratification to preserve class proportions.
- Oversampling minority classes in the training set to reduce the effect of class imbalance.

### Models

We explored two temporal modeling approaches:

- **BiLSTM + Multi-Head Attention:** two bidirectional LSTM layers learn motion patterns over time. A four-head self-attention layer and a dense classification head are used to predict the pain level. We tested different learning rates and batch sizes and selected the configuration using validation macro F1.
- **CNN + BiLSTM + Attention:** the second approach, described in the report, combines 1D convolutions for local motion patterns with BiLSTMs and a custom attention layer. It also uses a separate branch for the four survey features, along with additional statistical features and test-time augmentation.

The supplied notebooks implement the first approach in **TensorFlow/Keras**. The implementation of the second model is described in the report but is not included in the available code files.

## Results

The final challenge report gives the following approximate validation F1 scores:

| Model | Reported validation F1 |
| --- | ---: |
| BiLSTM + Multi-Head Attention | ~0.94 |
| CNN + BiLSTM + Attention | ~0.95 |

The saved output in the available BiLSTM notebook reports **0.8553 macro F1**, **0.9409 weighted F1**, and **93.98% accuracy** on its validation split. These are different metrics, so they should not be treated as the same score.

The best configuration in the notebook used a learning rate of `1e-3`, batch size `32`, and oversampling. The high-pain class remained the most difficult because it had relatively few examples.

## Repository files

- [`AN2DL_Challenge1_Code1.ipynb`](AN2DL_Challenge1_Code1.ipynb) — BiLSTM, multi-head attention, model selection, and submission generation.
- [`AN2DL_Challenge1_Code2.ipynb`](AN2DL_Challenge1_Code2.ipynb) — second uploaded notebook (currently identical to Code1).
- [`AN2DL_1_report.pdf`](AN2DL_1_report.pdf) — full project report covering both approaches and their results.

## Running the notebooks

The notebooks use **Python and TensorFlow/Keras** and were developed for a notebook environment such as Google Colab. To run them, you need the original challenge data:

- `pirate_pain_train.csv`
- `pirate_pain_train_labels.csv`
- `pirate_pain_test.csv`

Place these files in the working directory, or update the paths in the notebook. Run the cells in order to prepare the sequences, train and evaluate the model, and generate `submission.csv`. The dataset and trained weights are not included in the repository.

This is a course project and has not been validated for clinical pain assessment.
