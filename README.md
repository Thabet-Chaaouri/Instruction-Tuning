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
- [Finetune base Flan T5](https://www.philschmid.de/fine-tune-flan-t5) : To reproduce the same experiment I had to use the Trainer on a T4 free colab but with a gradient_checkpointing and a combination of train_batch_size=32, fp16 mixed precision (I didn't finish the training, but it is said that fp16 leads to overflow issues with T5). It took 4 hours of training duration for the T5 base model (which is 250M parameters large). I tried to run the same experiment on a colab pro with NVIDIA V100 but it didn't work. I suppose that the memory of the p3.2xlarge AWS EC2 Instance use in the tutorial is a little bit larger. [Colab notebook](https://colab.research.google.com/drive/1_RZgtC-_cZUCrInpsLQwmZrIY-ijvNFR?usp=sharing)
- [Finetune XL/XXL Flan T5 with deepspeed](https://www.philschmid.de/fine-tune-flan-t5-deepspeed) : The XXL FLAN T5 is an 11B model. there is no way to fully finetune the model in colab. the tutorial uses deepspeed on large EC2 instances with multiple GPUs.
It starts with the preprocessing outside of the GPU instance then loading the tokenized dataset from disk and the deepspeed config file in the train script like this :
![Screenshot](sdfrere.PNG)
Actually different experiments were conducted in this tutorial using different config deepspeed files:
![Screenshot](sdlkheio.PNG)
As fp16 can overflow, bf16 is the better choice because it provides significant advantages over fp32. We see also that it is better to keep a small batch size and not do offloading than the other way around. [Colab notebook](https://colab.research.google.com/drive/1Kl2ojG83-cWTip9-_hj_2mH7rTTBj5Pj?usp=sharing)
- [Finetune XXL Flan T5 with Lora](https://www.philschmid.de/fine-tune-flan-t5-peft) : In this tutorial, the 11B FLAN-T5 XXL was finetuned using Lora and 8bit quantization(bnb) from Peft.
The same techniques were used for data preprocessing.
The only differences was the construction of the peft model and the preparing for int8 training:
![Screenshot](Peft_training.PNG)
This configuration uses only 0.16% of the parameters of the model. The training took 10h and cost ~13.22$. A full fine-tuning on FLAN-T5-XXL with the same duration (10h) requires 8x A100 40GBs and costs ~322$.
During evaluation, the results were slightly better than a full finetuning of FLAN-T5 base. [Colab notebook](https://colab.research.google.com/drive/1S5L1HvYv61oVKH9aZZ26nJXNt4akivN7?usp=sharing)

Another notebook was created for comparing full fine-tuning and Peft lora fine-tuning on the base FLAN-T5. [Colab notebook](https://colab.research.google.com/drive/18EzRa2oSfjOQBYz1SnCnmjLMn-X4S01h?usp=sharing)

-Bloom:
- [Finetune 7B BLOOMZ with lora](https://www.philschmid.de/bloom-sagemaker-peft) : BLOOMZ 7b1 is the finetuned version from the pretrained [bloom 7b1](https://huggingface.co/bigscience/bloom-7b1). there is also a finetuned version for prompting in non english (ex: French) [bloomz-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt) a 7B parameters model, and a bigger one [bloomz-mt](https://huggingface.co/bigscience/bloomz-mt).

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
