# GPT_Playground
This projects implements a toy-example of GPT-2 with additional bells and whistles like several attention mechanisms, Mixture-of-Experts etc. 
To get started, this repo relies on [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) implementation. 

## Training data
To train a language model of GPT variant, this repo uses Harry Potter books. The dataset is already preprocessed, and it
can be found on [Kaggle](https://www.kaggle.com/datasets/moxxis/harry-potter-lstm).


## How to use?

This repo serves the purpose of quickly implementing the building blocks of SoTA LLMs for me. I have tried to make this repo useful for anyone
who wants to play around with building blocks of LLMs.

### Configs:
To instantiate and customize GPT variant `configs/model_configs.yaml` can be used. It allows to customize the attention mechanism, single/multi-token prediction,
a Dense LLM or a sparseMoE variant, an LLM with transfomer blocks/MAMBA/mixture of TX+MAMBA etc. 


## Progress and ToDo:
- [x] Implement vanilla transformer based LLM\
- [x] Implement trainings and evaluation pipeline\
- [x] Implement MoE blocks to convert Standard GPT into an Sparse MoE based language model\
- [x] Implement Multi-Query and Group-Query attention\
- [x] Implement multi-token prediction\
- [x] Implement evaluation mechanism (perplexity)\
- [x] Implement kv-caching\
- [ ] Implement MAMBA block as an alternative to regular transformer block and create e.g. small MAMBAFormer\
- [ ] Implement ROPE\
- [ ] Implement load balancing loss for MoE

