# Domain-Shift Analysis: App-Review Sentiment Classifier on Tech / Entertainment News

## Prediction distribution

The model produced predictions for all 1,033 tech and entertainment news articles. The distribution below shows how many times the classifier selected each label.

| Label | Count |
|---|---:|
| neutral | 457 |
| negative | 450 |
| positive | 126 |


This distribution matters because it shows whether the classifier behaves in a balanced way or whether it leans toward one class after moving into a new domain. Since the model was trained on app reviews, a strong skew toward one label would be a warning sign that the model is applying review-domain patterns to news articles.

## Confidence distribution

The predicted_probability column is the model confidence in its selected class.

| Metric | Value |
|---|---:|
| Mean predicted probability | 0.6183 |
| Median predicted probability | 0.5858 |
| Proportion above 0.90 | 0.0300 |
| Proportion below 0.60 | 0.5324 |

Confidence is especially important in a domain-shift setting. A model can be very confident even when it is wrong, because softmax probability measures the model's internal certainty, not real-world correctness. I would use low-confidence predictions as candidates for human review, and I would also sample high-confidence predictions to check for overconfidence.

## Five qualitative examples

The examples below were selected to show different model behaviors, including high-confidence cases and lower-confidence cases. They help reveal how an app-review sentiment model handles factual news prose.


### Example 1

- Article ID: NEWS_0892
- Excerpt: (CNN) -- It's a high-tech, high-stakes game of cat-and-mouse. Ryan Kelly, a Web designer from London, England, says his software was used to attack an Iranian Web site. As the Iranian government seeks
- Predicted label: negative
- Predicted probability: 0.9469
- Interpretation: This is a high-confidence prediction. It may look reliable, but under domain shift I would still inspect it because news text can be factual rather than opinion-based.

### Example 2

- Article ID: NEWS_0911
- Excerpt: LOS ANGELES, California (CNN) -- Rapper Kanye West and his business manager face vandalism, battery and grand theft charges in connection with a scuffle with photographers at Los Angeles International
- Predicted label: negative
- Predicted probability: 0.9431
- Interpretation: This is a high-confidence prediction. It may look reliable, but under domain shift I would still inspect it because news text can be factual rather than opinion-based.

### Example 3

- Article ID: NEWS_0909
- Excerpt: (CNN) -- Two words, delivered with index finger punctuating the air and directed at the president of the United States, made a little-known South Carolina congressman one of the most controversial men
- Predicted label: negative
- Predicted probability: 0.9429
- Interpretation: This is a high-confidence prediction. It may look reliable, but under domain shift I would still inspect it because news text can be factual rather than opinion-based.

### Example 4

- Article ID: NEWS_0167
- Excerpt: Ayaan Hirsi Ali is a Somalia-born writer, activist, and former member of the Dutch Parliament. She is an outspoken advocate for women's rights in Islamic society and a strong critic of Muslim extremis
- Predicted label: neutral
- Predicted probability: 0.3656
- Interpretation: This is a lower-confidence prediction. It shows that the model is less certain when the article language does not strongly match app-review language.

### Example 5

- Article ID: NEWS_1014
- Excerpt: LONDON, England -- A huge crowd gathered in London, UK on Friday for a mass "moonwalk" -- paying tribute to Michael Jackson by dancing to his most iconic songs and replicating his famous walk. Two you
- Predicted label: neutral
- Predicted probability: 0.3771
- Interpretation: This is a lower-confidence prediction. It shows that the model is less certain when the article language does not strongly match app-review language.


## Engineering judgment

I would not ship this model directly to production for tech or entertainment news sentiment classification without more validation. The model was trained on app reviews, which are short, opinion-heavy, and written by users. News articles are longer and often more factual. In production, this mismatch could create false positives where neutral reporting is labeled as emotional sentiment, or false negatives where negative news is missed because it does not sound like an app complaint. I would only use this as a baseline or triage tool with confidence thresholds and human review. A safer production system would need a labeled news-sentiment dataset, calibration checks, and likely fine-tuning on examples from the target news domain.
