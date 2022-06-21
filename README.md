# Transformers for legal contract review 👩‍⚖️ 📑
*The source code in this thesis was prepared at the Technical University of Denmark, Department of Applied Mathematics and Computer Science in fulfillment of the requirements for acquiring a Masters of Science degree in Human-Centered Artificial Intelligence. The project was prepared by Gustav Selfort Hartz. The workload of the project is 30 ECTS.*

**Project Abstract**

--insert here--



## Training

This training setup requires some slight altercations to the squad processor of transformers including a smaller chunksize for squad preprocessing and a different dataset structure. There is not provided a fork of transformer, but you can find the altered file in the root dir.
```
# Update squad file on ai platform GCP
cp squad.py /opt/conda/lib/python3.7/site-packages/transformers/data/processors/squad.py
```


### Custom questions 

See notebooks/Custom Questions Creation.ipynb

### Enviroment

Tested with the following python versions
Python 3.7.12
Python 3.8.13

The `conda_enviroment.py` file is from a local development enviroment and the training configuration can be extracted from the GCP vm image listed below.


***Setup GCP Enviroment used for training***
```bash
gcloud notebooks instances create contract2 --vm-image-project="deeplearning-platform-release"     --vm-image-name=pytorch-1-11-cu113-notebooks-v20220316-debian-10 --machine-type=n1-highmem-8 --location=europe-west4-a

# Adjust the data disk size

# Start the notebook 

# Use new altered transformers package
cp squad.py /opt/conda/lib/python3.7/site-packages/transformers/data/processors/squad.py

# Install packages not already there
pip install transformers
pip install pytorch-lightning

# Run training

```


## Paper remarks 

The thesis makes references to the code used for training transformers on SQuAD. This can be found from the commit history, but has been removed from the final repository to improve readability since the code was not pretty.

## Other notes

### Resources
Based on the following resources

- [Transformer docs](https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.BertModel)
- [Using a custom dataset and the preprocessing of squad](https://huggingface.co/transformers/v3.2.0/custom_datasets.html)
- [QA head](https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/bert/modeling_bert.py#L1792)

Enable GPU monitor - for maximizing GPU utilization - batch size etc.
https://cloud.google.com/compute/docs/gpus/monitor-gpus

### Git comments
This repo also has a function of syncing data between a remote server and my local instance, thus some commits might appear weird or not living up to good standards.

### Contributions
Have something you want to add just open a pull request :-)
