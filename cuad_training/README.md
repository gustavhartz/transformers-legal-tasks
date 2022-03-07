# bert-transformers-squad

Experimenting with custom Q&amp;A heads for bert models. Using pytorch lightning and multi GPU

Based on the following resources

- [Transformer docs](https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.BertModel)
- [Using a custom dataset and the preprocessing of squad](https://huggingface.co/transformers/v3.2.0/custom_datasets.html)
- [QA head](https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/bert/modeling_bert.py#L1792)

Enable GPU monitor - for maximizing GPU utilization - batch size etc.
https://cloud.google.com/compute/docs/gpus/monitor-gpus

### Optimizing the gpu utilization

The initial setup was 2x T4 gpus using a 'dp' strategy and a batch size of 8. This yielded around 50% utilization. To keep some overhead I will increase it to 14.

### Git comments

This repo also has a function of syncing data between a remote server and my local instance, thus some commits might appear weird or not living up to good standards.
