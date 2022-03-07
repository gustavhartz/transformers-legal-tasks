# bert-transformers-CUAD

Experimenting with custom Q&amp;A heads for bert models. Using pytorch lightning and multi GPU

Based on the following resources

- [Transformer docs](https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.BertModel)
- [Using a custom dataset and the preprocessing of squad](https://huggingface.co/transformers/v3.2.0/custom_datasets.html)
- [QA head](https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/bert/modeling_bert.py#L1792)

Enable GPU monitor - for maximizing GPU utilization - batch size etc.
https://cloud.google.com/compute/docs/gpus/monitor-gpus

### Optimizing the gpu utilization

The initial setup was 4x T4 gpus using a 'dp' strategy and a batch size of 8. This yielded around 50% utilization. To keep some overhead I will increase it to 14.

### Git comments

This repo also has a function of syncing data between a remote server and my local instance, thus some commits might appear weird or not living up to good standards.

---

## Cuad dataset for reproducable training - Transformer Tokenizer

> > More than 99% of the features generated from applying a sliding window to each contract do not contain any of the 41 relevant labels. If one trains normally on this data, models typically learn to always output the empty span, since this is usually the correct answer. To mitigate this imbalance, we downweight features that do not contain any relevant labels so that features are approximately balanced between having highlighted clauses and not having any highlighted clauses. For categories that have multiple annotations in the same document, we add a separate example for each annotation.

The issue with the data provided and the original code is that there is no script provided for creating this training data. Furthremore, there is no possibilities for experiments with how the splits were made to increase the amount of data.

This notebook helps create a dataset from data split presented in the original paper.

For the first basic version we will allow a few hyper paramaters to control the output data such as

- Stride type
- Context len
- Relationship between unanswerable questions and normal ones
- Max span - The span will be this size - a couple of tokens if we clip mid word
