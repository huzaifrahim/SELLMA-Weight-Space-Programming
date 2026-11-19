# Weight-Space Programming: Investigating Targeted Parameter Editing for Swedish Language Modelling

**Candidate:** Huzaif Rahim  
**Position:** AI Infrastructure Application Expert (SE.LLMA)  
**Institution:** Linköping University  
**Date:** 19 November 2026

## 1. Research Question and Motivation

This assignment investigates whether targeted modifications of neural-network parameters can influence language-specific behaviour while preserving performance on other languages.

The central research question is:

*Can a targeted parameter modification improve Swedish character prediction without substantially degrading English prediction?*

The hypothesis is that increasing the output biases associated with Swedish-specific characters (`å`, `ä` and `ö`) will improve their probability assignments in a shared Swedish-English character prediction model. A random parameter modification is included as a control.

The experiment is motivated by the concept of Weight-Space Programming (WSP), introduced by Takehi et al. (2026). The central idea is to treat neural-network weights as a program that can be inspected and modified directly, rather than relying exclusively on conventional optimisation methods.

## 2. Critical Assessment of the Original Research

In *Sparks of Weight-Space Programming: LLMs can natively read and write neural network weights*, Takehi, Teoh, Zhang and Khattab (2026) investigate whether a general-purpose large language model can interpret, modify and construct neural networks through direct interaction with their parameters.

The authors present eight experiments covering three main capabilities: reading neural-network weights, modifying existing networks and writing network parameters from scratch.

One of the strongest demonstrations is the identification of neural-network behaviour directly from model weights. For example, the authors report that GPT-6 Astra correctly identified the training games of four neural-network policies despite their having identical architectures and input-output dimensions. This illustrates the potential of interpreting learned behaviour from parameter structures rather than relying only on model outputs.

Another important experiment investigates the modification of a reinforcement-learning policy that failed at a physical control task. Astra identified a failure mechanism and modified the network parameters to achieve successful control behaviour.

Experiment 8 is particularly relevant to language-model applications. The authors investigate Qwen3.5-0.8B, identifying repetitive reasoning loops as a major failure mechanism. By inspecting and directly modifying selected MLP output weights in layers 15 and 19, Astra modified approximately 0.016% of the model's weights. The authors report that MATH-500 accuracy increased from 7.9% to 48.1%, while IFEval performance increased from 21.0% to 62.3%.

This is an interesting demonstration of targeted parameter intervention. However, the improvement should be interpreted carefully: the intervention addressed an existing reasoning failure rather than necessarily creating an entirely new reasoning capability.

The research also raises important questions about generalisation and interpretability. A successful modification on one architecture or task does not automatically establish that the same approach will work on other models.

An important distinction is the difference between reasoning directly about weights and feedback-driven optimisation. In the former, an LLM attempts to identify meaningful structures or failure mechanisms in the parameters and predict the effects of modifications. In the latter, candidate modifications are assessed through repeated model executions and feedback. These approaches can complement one another, but successful optimisation through feedback does not by itself demonstrate semantic understanding of individual weights.

Overall, the paper presents promising early evidence that LLMs can interact with neural-network parameters in ways that extend beyond conventional prompt engineering and training workflows. Nevertheless, reproducibility, generalisation and the preservation of unrelated capabilities remain important questions for further investigation.

## 3. Experimental Methodology

### 3.1 Model and Dataset

A lightweight character-context prediction model was implemented using Python and NumPy.

The model uses the three preceding characters as input, represented through one-hot encoding, followed by a linear softmax output layer.

A manually curated dataset containing ten Swedish and ten English sentences was used. The sentences cover general language usage, research and scientific terminology.

Both languages were combined during training using a shared character vocabulary.

The dataset was shuffled using random seed 42 and divided into:

- Training: 60% (six sentences per language).
- Validation: 20% (two sentences per language).
- Testing: 20% (two sentences per language).

The model was trained jointly on both languages for 300 epochs.

### 3.2 Experimental Conditions

Three conditions were evaluated.

**1. Baseline**

The original trained model without parameter modification.

**2. Targeted parameter edit**

An additive modification was applied to the output-bias parameters associated with the Swedish-specific characters `å`, `ä` and `ö`.

The modification strength was selected using Swedish validation cross-entropy. The selected value was 1.2.

**3. Random control**

A random output-bias modification was generated with the same L2 magnitude as the targeted modification, but with a random direction.

The model weight matrix was kept unchanged across all three conditions. Only the output-bias vector was modified.

### 3.3 Evaluation Metrics

The following metrics were used:

- **Next-character accuracy:** The proportion of correctly predicted next characters.
- **Cross-entropy:** Measures the quality of the predicted character probabilities. Lower values indicate better performance.
- **Swedish diacritic recall:** Measures the proportion of actual occurrences of `å`, `ä` and `ö` that were correctly predicted.

The validation data was used to select the targeted edit strength. Final comparisons were performed using the held-out test sentences.

## 4. Experimental Results

The following results were obtained from the locally executed experiment.

| Condition | Swedish accuracy | Swedish cross-entropy | English accuracy | English cross-entropy |
|---|---:|---:|---:|---:|
| Baseline | 21.62% | 2.8721 | 20.75% | 2.6282 |
| Targeted edit | 21.62% | 2.8034 | 20.75% | 2.6672 |
| Random edit | 25.23% | 2.9163 | 21.70% | 2.6327 |

Swedish-specific character recall was zero in all three conditions.

### 4.1 Targeted Modification

The targeted edit reduced Swedish test cross-entropy from 2.8721 to 2.8034, while next-character accuracy remained unchanged at 21.62%.

English cross-entropy increased from 2.6282 to 2.6672, while accuracy remained unchanged.

This indicates that the targeted modification changed the probability distribution of the model's predictions. However, it did not demonstrate an improvement in Swedish character prediction accuracy or diacritic recall.

### 4.2 Random Control

The random modification produced higher next-character accuracy for both languages. However, its cross-entropy was worse than the baseline for both languages.

This illustrates why accuracy alone is insufficient for evaluating parameter modifications. Improvements in one metric do not necessarily indicate an overall improvement in model behaviour.

### 4.3 Regression Analysis

The targeted modification produced a reduction in Swedish cross-entropy accompanied by an increase in English cross-entropy.

This suggests a possible trade-off between the two languages. However, the very small test set does not support a statistically reliable conclusion.

Neither the targeted nor random modification improved Swedish-specific character recall.

## 5. Interpretation and Limitations

The experiment demonstrates that a manually selected parameter modification can change the output probability distribution of a character-context prediction model.

However, the results do not establish that Weight-Space Programming improves Swedish language capabilities.

Several limitations must be considered:

1. **Dataset size:** Only twenty manually curated sentences were used, which is insufficient for general language-modelling conclusions.
2. **Test-set size:** Only two sentences per language were reserved for testing.
3. **Model architecture:** The experiment uses a linear character-context softmax classifier, not a transformer or large language model.
4. **Parameter intervention:** The modification was applied to output-bias parameters rather than internal transformer weight matrices.
5. **Manual intervention:** The Swedish-specific editing direction was predefined rather than independently discovered through weight interpretation.
6. **Randomness:** Only one random seed and one random control direction were evaluated.
7. **Generalisation:** The experiment does not establish whether the observed changes generalise to unseen domains, longer texts or other languages.

**Computational constraints:** The experiment was conducted on a MacBook Air (13-inch, 2017) equipped with a 1.8 GHz dual-core Intel Core i5 processor and 8 GB RAM. These computational constraints influenced the choice of a lightweight NumPy-based architecture and CPU-based implementation rather than a large transformer model. The experiment therefore serves as a computationally accessible proof of concept rather than a benchmark of large-scale weight-space editing. Future investigations using GPU-enabled hardware would allow evaluation on open-weight transformer models.

Consequently, this experiment should be interpreted as a preliminary demonstration of output-bias intervention rather than a reproduction of the original Weight-Space Programming approach.

## 6. Recommendations for SE.LLMA

The experiment highlights the importance of evaluating targeted parameter modifications against both an unchanged baseline and random controls.

A more rigorous investigation should use an open-weight multilingual transformer and substantially larger, comparable Swedish and English datasets.

Future experiments should investigate:

- Whether an LLM can independently identify relevant parameters associated with a particular behavioural failure.
- Whether repeated forward-pass feedback improves the quality of parameter modifications.
- Whether targeted edits outperform random edits and conventional fine-tuning.
- Whether improvements transfer to unseen datasets and different language tasks.
- Whether targeted modifications preserve performance on unrelated capabilities.

Multiple random seeds, larger held-out test sets and systematic regression benchmarks should be included.

For practical AI infrastructure applications, parameter editing should also be accompanied by reproducibility checks, parameter-difference tracking and automated regression testing.

The central challenge is not simply to modify weights, but to identify meaningful interventions whose effects are interpretable, measurable and reproducible.

## 7. AI Tools and Contributions

ChatGPT was used as a research and coding assistant to support experimental design, Python implementation, debugging and report preparation.

The Python experiment was executed locally by the candidate. The numerical results reported here were obtained from the execution outputs and saved experiment files.

The candidate reviewed the methodology, experimental results and scientific interpretation. No unexecuted experiments or fabricated measurements are presented.

## References

Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026). *Sparks of Weight-Space Programming: LLMs can natively read and write neural network weights*. MIT CSAIL, 2 November 2026.

https://rikiyat.github.io/blog/weight-space-programming/