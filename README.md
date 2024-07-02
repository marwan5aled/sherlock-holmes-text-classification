# Sherlock Holmes Text Classification

This project classifies sentences from the text of "Sherlock Holmes" into different categories based on the presence of Sherlock or Watson.

## Requirements

- pandas
- numpy
- re
- os
- requests
- scikit-learn

## Usage

1. Ensure you have the required libraries installed:
    ```bash
    pip install pandas numpy requests scikit-learn
    ```

2. Run the script:
    ```bash
    python text_classification.py
    ```

## Description

- The script downloads the text of "Sherlock Holmes" from Project Gutenberg if it doesn't already exist locally.
- It processes the text and splits it into sentences.
- It checks for an annotation CSV file, which labels sentences based on the presence of Sherlock and Watson.
- The script trains a text classification model using these annotations and evaluates its performance.

## Output

- The script prints the classification accuracy and a detailed classification report.

