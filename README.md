# transformers-legal-tasks
Investigation and modelling of transformers for the use in tasks such as legal contract element extraction. Part of master thesis in Human-Centered Artificial Intelligence from DTU


This training setup requires some slight altercations to the squad processor of transformers including a smaller chunksize for squad preprocessing and a different dataset structure. There is not provided a fork of transformer, but you can find the altered file in the root dir.
```
# Update squad file on ai platform GCP
cp squad.py /opt/conda/lib/python3.7/site-packages/transformers/data/processors/squad.py
```

### Enviroment

Tested with the following python versions
Python 3.7.12
Python 3.8.13
