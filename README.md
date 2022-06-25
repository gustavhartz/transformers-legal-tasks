# Transformers for legal contract review 👩‍⚖️ 📑
*The source code in this thesis was prepared at the Technical University of Denmark, Department of Applied Mathematics and Computer Science in fulfillment of the requirements for acquiring a Masters of Science degree in Human-Centered Artificial Intelligence. The project was prepared by Gustav Selfort Hartz. The workload of the project is 30 ECTS.*

**Project Title**
*Exploring CUAD using RoBERTa span-selection QA models for legal contract review*

**Project Abstract**
Deep learning models for span-selection question and answering have, with the advent of transformer models such as BERT and RoBERTa, shown excellent performance on established tasks like SQuAD and GLUE. Training these models requires large annotated datasets, making them expensive to use in highly specialized domains.

The Contract Understanding Atticus Dataset (CUAD) is a new public dataset by The Atticus Project, Inc., suitable for training span-selection models for legal contract review. This thesis presents an in-depth analysis of the performance obtained from the RoBERTa models presented by the authors of CUAD. It examines the models' robustness to multiple different new formulations of the original $41$ questions, in addition to methods for reducing the number of model parameters by upwards of 20\% and model inference time by close to 50\%, with little to no effect on performance. Moreover, ways of boosting model performance based on a thorough analysis of the CUAD dataset in terms of annotation quality, question category annotation distribution, and potentially unmarked answers were tested. Additionally, a new proposal for pre-processing CUAD is presented to mitigate ambiguity in training data. The thesis also argues for using other evaluation metrics than CUAD. These new metrics show that performance across all $41$ question categories is better than indicated in the original paper and likely usable for a human-in-the-loop legal contract review system.

The thesis improves ***top 3 Has Ans metric*** from 92.5% to 94.1%, but cannot conclude that this is due to augmentations made to the dataset. Additionally, it suggests that performance gains can be found by adjusting objective functions to suit the performance metrics better.

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

## Using it on a contract example
```python
# Requirements
from transformers import pipeline
import PyPDF2

pipe = pipeline("question-answering", model="Rakib/roberta-base-on-cuad", device=0)

# The contract to investigate 
pdf_file = open('data/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/LinkPlusCorp_20050802_8-K_EX-10_3240252_EX-10_Affiliate Agreement.pdf', 'rb')

pdfReader = PyPDF2.PdfFileReader(pdf_file)
num_pages = pdfReader.numPages
count = 0
# Read each page
texts = ""
while count < num_pages:
  pageObj = pdfReader.getPage(count)
  text = pageObj.extractText()
  texts+= "\n" + text
  count += 1

# Format data and add question
context = texts
question = "Highlight the parts (if any) of this contract related to \"Document Name\" that should be reviewed by a lawyer. Details: The name of the contract"

# Get a pred using the V2 pred logic
pipe(question = question, context = context, top_k=3)

```
## Other notes

### Citation

If you found the code of thesis helpful you can please cite it :)

```
@thesis{ha2022,
  author = {Hartz, Gustav Selfort},
  title = {Exploring CUAD using RoBERTa span-selection QA models for legal contract review},
  language = {English},
  format = {thesis},
  year = {2022},
  publisher = {DTU Department of Applied Mathematics and Computer Science}
} 
```

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
