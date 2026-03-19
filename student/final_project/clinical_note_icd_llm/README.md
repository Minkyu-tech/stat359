# Comparing General vs Biomedical BERT for ICD Classification

## 1. Overview and Objective

This project studies whether **domain-specific pretraining** improves clinical text classification for ICD prediction.

The task is to predict one of the **Top-30 ICD-10 three-character categories** from processed clinical notes derived from the Hugging Face dataset ```rntc/mimic-icd-reformulations-medgemma-27b-text-it-2```. The overall project follows the course handout’s open-ended option of fine-tuning a Hugging Face model for a custom task.

The main objective is twofold:

1. Compare a **general-domain BERT** and a **biomedical-domain BERT** on the same ICD classification task.
2. Test whether **Weighted Cross-Entropy** improves performance under class imbalance.

This project is designed to be focused, reproducible, and suitable for a README or technical report, which matches the course emphasis on depth, methodology, and clean experimental design.

---

## 2. Research Questions and Hypotheses

### RQ1. Does biomedical-domain pretraining improve ICD classification performance compared to general-domain pretraining?

**Hypothesis 1:**  
```microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext``` will outperform ```google-bert/bert-base-uncased``` because biomedical pretraining should better capture medical terminology and clinical language patterns.

### RQ2. Does Weighted Cross-Entropy improve performance over standard Cross-Entropy under imbalanced class distributions?

**Hypothesis 2:**  
Weighted Cross-Entropy will improve **Macro F1**, especially for lower-frequency classes, even if overall accuracy changes only slightly.

### Supporting Hypothesis

**Hypothesis 3:**  
Differences in tokenizer behavior between BERT and BiomedBERT may partially explain performance differences, especially for biomedical vocabulary.

---

## 3. Model Selection and Rationale

### Model 1: General-Domain Baseline
```google-bert/bert-base-uncased```

This model serves as a strong baseline because it is a standard pretrained transformer widely used for text classification. It represents a general English language model without biomedical specialization.

### Model 2: Biomedical-Domain Model
```microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext```

This model is chosen because it is pretrained on biomedical text, making it more appropriate for clinical notes and medical terminology.

### Loss Functions
Two loss settings are used:

- **Standard Cross-Entropy**
- **Weighted Cross-Entropy**

Weighted CE is computed as:

$$Weight_i = \frac{N}{C \times Count_i}$$


where
- \($N$\): the total number of training samples
- \($C$\): the number of classes = 30
- \($Count_i$\): the number of samples in class \($i$\)

---

## 4. Data and Preprocessing Strategy

### Dataset

The source dataset is:

```rntc/mimic-icd-reformulations-medgemma-27b-text-it-2```

The preprocessing pipeline keeps only the relevant fields:

- ```reformulation```
- ```icd_code```

Then it extracts clinically informative note sections such as:

- Chief Complaint / Reason for Visit
- History of Present Illness / History
- Hospital Course / Treatment

The course section is truncated to a maximum length, and very short notes are removed.

### ICD Labeling Strategy

Original ICD codes are converted into **3-character ICD-10 major categories**.  
For example:

```I21.3 -> I21```  
```E11.9 -> E11```

The top 30 most frequent categories are selected, then sampled proportionally to preserve the original imbalance pattern. A fixed label mapping is created and saved as JSON files.

### Train/Validation/Test Split

The final dataset is split with stratification:

```Train: 80%```  
```Validation: 10%```  
```Test: 10%```

This keeps class proportions consistent across splits.

---

## 5. Training Configuration

The current recommended configuration for the local environment is:

```max_length = 256```  
```train_batch_size = 8```  
```eval_batch_size = 16```  
```learning_rate = 2e-5```  
```epochs = 5```  
```early_stopping_patience = 2```  
```weight_decay = 0.01```  
```warmup_ratio = 0.1```  
```fp16 = True```

### Optimizer Choice

**AdamW** is the most appropriate default optimizer here.  
It is the standard choice for BERT-style fine-tuning because it handles weight decay properly and is well-supported in the Hugging Face training stack. It is not guaranteed to be universally optimal, but for this project it is the **best practical and reliable choice**.

### Tokenization

Each model uses its own pretrained tokenizer with:

```padding = True```  
```truncation = True```  
```max_length = 256```

Tokenizer outputs will also be compared qualitatively to analyze biomedical term preservation.

---

## 6. Evaluation Metrics and Expected Outcomes

### Primary Metric
- **Macro F1**

This is the most important metric because the class distribution is still imbalanced even after Top-30 filtering and proportional sampling.

### Secondary Metrics
- **Accuracy**
- **Hit@3**
- **Hit@5**

Hit@3 and Hit@5 are useful because they show whether the correct ICD code appears among the model’s top-ranked predictions, which is meaningful for medical code recommendation settings.

### Additional Analysis
- Confusion Matrix
- Per-class F1
- Tokenizer comparison examples

### Expected Results

- **BiomedBERT + CE** is expected to outperform **BERT-base + CE** on Macro F1.
- **BiomedBERT + Weighted CE** is expected to improve Macro F1 further, especially for minority classes.
- Accuracy may not improve as much as Macro F1, which would still be an acceptable and interpretable result.

---

## 7. Planned Experiments

### Experiment 1
```BERT-base + Cross-Entropy```

Purpose: establish a general-domain baseline.

### Experiment 2
```BiomedBERT + Cross-Entropy```

Purpose: test whether biomedical pretraining improves ICD classification.

### Experiment 3
```BiomedBERT + Weighted Cross-Entropy```

Purpose: test whether imbalance-aware training improves performance beyond standard CE.

---

## 8. Project Significance

This project contributes a focused comparison between **general-domain** and **biomedical-domain** transformer fine-tuning for ICD prediction. It also evaluates a practical imbalance-handling strategy using Weighted Cross-Entropy.

Because the project combines model comparison, domain adaptation, preprocessing design, and evaluation analysis, it fits the course expectation for a technically meaningful and well-scoped open-ended LLM project.