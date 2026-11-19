# Weight-Space Programming: Targeted Parameter Editing

**Candidate assignment:** AI Infrastructure Application Expert (SELLMA)  
**Institution:** Linköping University  
**Date:** 19 November 2026

## 1. Project overview

This project investigates a small, controlled parameter-editing experiment inspired by weight-space programming.

The experiment tests whether a targeted modification to a model's output parameters can improve its handling of Swedish-specific characters (`å`, `ä`, and `ö`) while monitoring its performance on English text.

This is a pilot experiment using a lightweight character-level model. It is not a replication of the large language model experiments described in the reference paper.

## 2. Research question

Can a targeted change to the output-bias parameters of a character-level language model reduce Swedish test cross-entropy, and what effect does this intervention have on English performance?

Three conditions are compared:

1. **Baseline:** the trained model without an intervention.
2. **Targeted edit:** an increase to the output biases associated with `å`, `ä`, and `ö`.
3. **Random edit:** a random output-bias modification with the same L2 magnitude as the targeted edit.

The edit strength is selected using Swedish validation data. Final results are measured on held-out test sentences.

## 3. Model and data

The model is a linear character-context softmax model implemented in NumPy.

- Context length: 3 preceding characters
- Training epochs: 300
- Learning rate: 0.15
- Random seed: 42
- Languages: Swedish and English
- Dataset: 10 manually curated sentences per language

The sentences are split by sentence into training, validation and test sets:

- 6 training sentences per language
- 2 validation sentences per language
- 2 test sentences per language

A shared character vocabulary is constructed from the corpus.

The model is deliberately small so that the experiment can run on a standard CPU without specialised hardware.

## 4. Experimental procedure

The model is trained jointly on the Swedish and English training sentences.

The targeted intervention adds a scalar multiple of a direction vector to the output-bias vector. The direction vector assigns a value of one to the Swedish-specific characters `å`, `ä`, and `ö`, and zero to all other characters.

The intervention strength is selected from 21 candidate values between 0 and 2 using Swedish validation cross-entropy.

A random control intervention is also generated. Its direction is random, but its L2 magnitude is matched to that of the selected targeted edit.

The final comparison uses the held-out test sentences.

**Important:** the intervention modifies the output-bias vector, not the model's weight matrix. The experiment therefore examines targeted output-parameter editing rather than a complete implementation of weight-space programming.

## 5. Results

| Condition | Swedish accuracy | Swedish cross-entropy | Swedish diacritic recall | English accuracy | English cross-entropy |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.2162 | 2.8721 | 0.0000 | 0.2075 | 2.6282 |
| Targeted edit | 0.2162 | 2.8034 | 0.0000 | 0.2075 | 2.6672 |
| Random edit | 0.2523 | 2.9163 | 0.0000 | 0.2170 | 2.6327 |

The selected targeted edit strength was 1.2.

The targeted edit reduced Swedish test cross-entropy from 2.8721 to 2.8034, while next-character accuracy remained unchanged. However, Swedish diacritic recall remained zero.

English cross-entropy increased from 2.6282 to 2.6672 after the targeted edit. This indicates a small regression in the English cross-entropy measure.

The random control increased Swedish accuracy but also increased Swedish cross-entropy. This illustrates why accuracy and cross-entropy should be considered together.

## 6. Interpretation and limitations

The results provide limited evidence that a targeted output-bias intervention can change model behaviour. The reduction in Swedish cross-entropy is modest and does not translate into improved accuracy or successful prediction of Swedish diacritics.

The experiment has substantial limitations:

- The dataset contains only 20 sentences.
- The held-out test set contains only two sentences per language.
- The model is a simple linear character-context model, not a transformer or large language model.
- The intervention modifies output biases rather than internal weight matrices.
- Only one random seed and one random control intervention were evaluated.
- The small test set makes the results sensitive to individual sentences.

Consequently, these findings should be interpreted as a proof-of-concept pilot, not as evidence that targeted weight editing reliably produces the intended behaviour.

## 7. Reproducibility

Use Python 3.11 or a compatible Python version.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the dataset:

```bash
python src/check_data.py
```

Run the experiment:

```bash
python src/experiment.py
```

The experiment writes the following files to the `experiments/` directory:

- `results.json` — evaluation metrics and experiment configuration
- `model_weights.npz` — trained model parameters and edited output biases

The random seed is set to 42 in the experiment script.

## 8. Project structure

```text
SELLMA_Weight_Space_Programming/
├── data/
│   └── corpus.json
├── experiments/
│   ├── config.json
│   ├── model_weights.npz
│   └── results.json
├── src/
│   ├── check_data.py
│   ├── experiment.py
│   └── make_pdf.py
├── .gitignore
├── README.md
├── requirements.txt
├── research_note.md
└── report.pdf
```

## 9. Use of AI tools

ChatGPT was used to assist with experimental planning, code development, debugging and editing of the research note. The experiment was executed locally, and the resulting outputs were reviewed and interpreted by the candidate.

## 10. Reference

Takehi et al. (2026). *Sparks of Weight-Space Programming: LLMs can natively read and write neural network weights.*

The experiment in this repository is an independent, small-scale pilot inspired by the research direction discussed in this paper.
