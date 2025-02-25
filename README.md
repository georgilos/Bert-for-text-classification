This repository is part of my thesis on training a LLM like BERT for hatespeech classification using active learning techniques

Dropbox link: https://www.dropbox.com/scl/fo/brj59vrixis5rkz5182j3/h?rlkey=nnhu2eoh48wutke73ezntvs3w&st=x5iiz72a&dl=0

## Methods presented:
- Active learning with outliers and low confidence (**Active+BERT.ipynb**)
- Support pair active learning with Hybrid loss (**test.py**)

## Execution (for the test.py):
- Before running, adjust the **max_iterations** and **n** (samples) variables 
- Run the file and follow the prompted instructions
- The model's weights, final cluster centroids and their classification labels will be saved in the **/models** directory
- To generate labels for new unseen texts (**unseen_texts.csv**), run the  inference.py file

### The "Active+Bert.ipynb" notebook uses Active learning techniques from Robert Monarch's repository: https://github.com/rmunro/pytorch_active_learning.git 

### The "test.py" uses the method presented in the paper https://arxiv.org/abs/2204.10008 and the constrained DBSCAN algorithms from the paper  https://link.springer.com/chapter/10.1007/978-3-540-72530-5_25
