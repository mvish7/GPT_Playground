# GPT_Playground
This projects implements a toy-example of GPT-2 with additional bells and whistles like several attention mechanisms, Mixture-of-Experts etc. 
To get started, this repo relies on [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) implementation. 

## Training data
To train a language model of GPT variant, this repo uses Harry Potter books. The dataset is already preprocessed, and it
can be found on [Kaggle](https://www.kaggle.com/datasets/moxxis/harry-potter-lstm).


ToDo:\
[x] Implement MoE blocks to convert Standard GPT into an Sparse MoE based language model\
[x] Implement Multi-Query and Group-Query attention\
[ ] Implement multi-token prediction\
[ ] Implement MAMBA block as an alternative to regular transformer block\
[ ] Implement evaluation mechanism (perplexity)
