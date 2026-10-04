# 🌍 LinguaScope

Machine learning system for multilingual language identification and comparative linguistic analysis.

## Features

- Detects the language of any text (20 languages, including Hindi, Urdu, Arabic and Japanese)
- Shows the top 5 predictions with confidence, script, and language family
- Compares two texts using character n-gram profiles
- Batch detection: upload a CSV and download the results
- Builds a language similarity heatmap and a language tree (dendrogram)
- Streamlit web interface

## Method

Text cleaning → TF-IDF character n-grams (1 to 3) → Logistic Regression and Naive Bayes.
Short-text data augmentation was added because the dataset has mostly long reviews.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | <<< 00.00% >>> |
| Naive Bayes | <<< 00.00% >>> |
| Short text (1 to 6 words) | <<< 00.00% >>> |

## Screenshots

![Demo](screenshots/demo.png)
![Language tree](outputs/language_tree.png)
![Heatmap](outputs/similarity_heatmap.png)
![Confusion matrix](outputs/confusion_matrix.png)

## How to run

```bash
pip install -r requirements.txt
python get_data.py
python linguascope.py
streamlit run app.py
```

To run the tests:

```bash
pip install pytest
pytest
```

## Project structure

| File | Purpose |
|---|---|
| `get_data.py` | Downloads the dataset |
| `linguascope.py` | Trains the models and creates the charts in `outputs/` |
| `app.py` | Streamlit web app |
| `tests/` | Basic tests for the model |
| `outputs/` | Saved model, charts and statistics |

## Limitations

- Very short texts (1 to 2 words) are less accurate.
- Mixed-language text such as Hinglish is not handled yet.
- Only 20 languages are supported.

## Dataset

[papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification) on Hugging Face.

## Future work

- Indian languages: Bhojpuri, Maithili, Magahi
- Hinglish detection
- fastText or BERT comparison

## Author

Juhi Sharma