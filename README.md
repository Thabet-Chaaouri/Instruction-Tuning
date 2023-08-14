# Instruction-Tuning

Unsupervised pretraining such as Causal Language Modeling

Instruction finetuning (Multi-task finetuning)

Alignment with RLHF

### Adapting to my single task

1 - Unsupervised finetuning on a specific task (an input-output schema) of an instruction tuned LLM.

[Unsupervised finetuning BLOOM](https://colab.research.google.com/drive/1ARmlaZZaKyAg6HTi57psFLPeh0hDRcPX?usp=sharing)

2 - Instruction finetuning on a single task of an instruction tuned LLM

[instruction Tuning Falcon 7b](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing#scrollTo=OCFTvGW6aspE)

risks : Catastrophic forgetting

solutions : Use PEFT to finetune only a set of parameters and freeze the original ones
