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

### Plan

-FlanT5:
- [Finetune base Flan T5](https://www.philschmid.de/fine-tune-flan-t5) : used Trainer on T4 free colab with a combination of train_batch_size=32, gradient_checkpointing, fp16 mixed precision and it takes 4 hours of training duration. T5 base is 250M model
- [Finetune XL/XXL Flan T5 with deepspeed](https://www.philschmid.de/fine-tune-flan-t5-deepspeed)
- [Finetune XXL Flan T5 with Lora](https://www.philschmid.de/fine-tune-flan-t5-peft)

-Bloom:
- [Finetune 7B BLOOMZ with lora](https://www.philschmid.de/bloom-sagemaker-peft)

-LLAMA2:
- [Ressources about LLama2](https://www.philschmid.de/llama-2)
- [Finetune LLAMA2 using Qlora](https://www.philschmid.de/sagemaker-llama2-qlora)
- [Instruction tuning LLama2 with trl and SFTTrainer](https://www.philschmid.de/instruction-tune-llama-2)
- [Deploying LLama2 on Sagemaker](https://www.philschmid.de/sagemaker-llama-llm)

-Falcon:
- [Finetune Falcon with QLora](https://www.philschmid.de/sagemaker-falcon-qlora)
- [Finetune Falcon using Qlora with Flash Attention](https://www.philschmid.de/sagemaker-falcon-180b-qlora)
- [Finetune Falcon with deepspeed and Lora](https://www.philschmid.de/deepspeed-lora-flash-attention)


-Compare GPTQ and bnb:
- [GPTQ and Hugging Face Optimum](https://www.philschmid.de/gptq-llama)
