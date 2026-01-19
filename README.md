# Opioid-induced-respiratory-prediction-using-encoders

This repository implements a machine learning pipeline designed to predict **respiratory depression** using clinical data. The project employs a unique approach by converting structured electronic health record (EHR) data into natural language clinical narratives, which are then used to fine-tune specialized clinical transformer models (Encoders).

> **Note on Data:** The dataset used in this project is an in-house HMC (Hamad medical corporation) dataset. Due to privacy and HIPAA regulations, the raw data has been excluded from this repository.

## Project Overview

The pipeline consists of two primary stages:

1. **Narrative Generation**: Transforming tabular patient data (vitals, medications, comorbidities, and surgical history) into cohesive, anonymized clinical prose.

2. **Encoder Fine-Tuning**: Utilizing clinical-specific BERT architectures to classify these narratives and predict the risk of respiratory depression.

## File Descriptions

### 1. `Narrative.ipynb`

This notebook handles the initial data engineering and text synthesis.

- **Data Cleaning**: Anonymizes patient records by converting absolute dates into "days since admission" and stripping identifiers like `encntr_id` and `fin`.
- **Narrative Synthesis**: Contains a robust transformation engine (`generate_clinical_narrative`) that converts complex medical features—such as BMI, smoking status, perioperative opioid dosages (Fentanyl, Morphine, Benzodiazepines), and surgical durations—into a detailed clinical narrative for each patient.
- **Data Serialization**: Outputs a processed pickle file (`my_project_data.pkl`) containing the text-based representations ready for deep learning.

### 2. `Encoder.ipynb`

This notebook implements the deep learning and evaluation pipeline.

- **Model Benchmarking**: Fine-tunes and compares multiple state-of-the-art clinical encoders, including `MedEmbed-large-v0.1`, `Bio_ClinicalBERT`, and `medicalai/ClinicalBERT`.
- **Training Pipeline**: Manages stratified data splitting (60/20/20), tokenization using Hugging Face `AutoTokenizer`, and hyperparameter tuning using the `Trainer` API.
- **Metrics & Evaluation**: Calculates AUROC, F1-Score, Precision, and Recall specifically for imbalanced clinical classes.
- **Embedding Extraction**: Extracts `[CLS]` token embeddings from the base models to preserve features for future interpretability studies.

## Requirements

To run this code, you will need the following Python libraries:

### Python Packages
- `pandas`: For data manipulation and processing of clinical records.
- `numpy`: For numerical operations and array handling.
- `scikit-learn`: For dataset splitting and calculating evaluation metrics (AUROC, F1, etc.).
- `torch`: For deep learning model execution using PyTorch.
- `transformers`: For accessing Hugging Face pre-trained clinical models and the `Trainer` API.
- `datasets`: For efficient data loading and mapping within the Hugging Face ecosystem.
- `tqdm`: For progress bars during model training and embedding extraction.
- `openpyxl`: Required for reading the initial Excel-based clinical datasets.

### Installation Instructions

You can install all required Python libraries using pip:

```bash
pip install pandas numpy scikit-learn torch transformers datasets tqdm openpyxl
