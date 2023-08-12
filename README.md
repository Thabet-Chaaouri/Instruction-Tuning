# Instruction-Tuning

Unsupervised pretraining such as Causal Language Modeling

Instruction finetuning (Multi-task finetuning)

Alignment with RLHF

### Adapting to my single task

1 - Unsupervised finetuning on a specific task (an input-output schema) of an instruction tuned LLM.

2 - Instruction finetuning on a single task of an instruction tuned LLM

risks : Catastrophic forgetting

solutions : Use PEFT to finetune only a set of parameters and freeze the original ones
