# Module 7 Week A — Integration Task: Domain-Shift Analysis

This project applies my fine-tuned app-review sentiment classifier from Lab 7A to a different text domain: tech, entertainment, and digital-culture news articles. The purpose is to test how a model trained on one kind of writing behaves when it is moved into a new domain.

## Hugging Face Hub model

My model is hosted on Hugging Face Hub here:

https://huggingface.co/omarallahham/m7-app-review-sentiment

The model id used by the pipeline is:

omarallahham/m7-app-review-sentiment

## Reproducing the run locally

To reproduce the full prediction file, install the dependencies and configure the model id:

pip install -r requirements.txt
cp .env.example .env

Then edit .env and set:

MODEL_HUB_ID=omarallahham/m7-app-review-sentiment

After that, run:

make apply

On Windows PowerShell, if make is not installed, the same run can be reproduced with:

$env:MODEL_HUB_ID="omarallahham/m7-app-review-sentiment"
Remove-Item Env:CORPUS_PATH -ErrorAction SilentlyContinue
Remove-Item Env:OUTPUT_PATH -ErrorAction SilentlyContinue
python apply.py

The expected output is predictions.csv. This file contains 1,033 rows and the columns article_id, text_excerpt, predicted_label, and predicted_probability.

## Smoke test

The smoke test uses a small five-row fixture and a public substitute model. It checks that the pipeline runs without depending on my specific model.

make smoke

PowerShell version:

$env:MODEL_HUB_ID="distilbert-base-uncased-finetuned-sst-2-english"
$env:CORPUS_PATH="data/tech_news_articles_smoke.csv"
$env:OUTPUT_PATH="predictions_smoke.csv"
python apply.py

## What the model was trained on

The model was fine-tuned on app-review sentiment data. That domain usually contains short user-written comments with clear opinions. App reviews often include direct emotional language, such as praise, complaints, frustration, satisfaction, or requests for improvement. Because of this, the model learned sentiment patterns that are common in review-style text.

In this task, I apply the same model to tech and entertainment news. This is a different domain because news articles are usually longer, more factual, and less directly emotional than app reviews. A news article may describe a company update, product launch, media event, or legal issue without expressing a clear positive or negative opinion. This creates domain shift.

## Why this assignment matters

The important engineering question is whether a model trained on one domain can be trusted in another domain. Even if the code runs and the model returns labels for every article, the predictions may not represent real news sentiment. The model may force factual reporting into positive, neutral, or negative categories because those are the only labels it knows.

This assignment helps identify where the classifier generalizes and where it breaks. It also shows why confidence scores must be interpreted carefully. A high softmax probability does not always mean the prediction is correct, especially when the input text comes from a domain that differs from the training data.

## Submission checklist

Before submitting, I verified that apply.py loads the model from Hugging Face Hub, reads the model id from MODEL_HUB_ID, reads labels from model.config.id2label, and generates predictions.csv for the full corpus.

## License

This repository is provided for educational use only. See LICENSE for terms.
