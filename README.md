# Instruction-Tuning

Unsupervised pretraining such as Causal Language Modeling

Supervised Instruction finetuning (Multi-task finetuning)

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

FLAN fine-tunes T5 on a large set of varied instructions that use a simple and intuitive description of the task, such as “Classify this movie review as positive or negative,” or “Translate this sentence to Danish.”. The paper shows that by training a model on these instructions it not only becomes good at solving the kinds of instructions it has seen during training but becomes good at following instructions in general. It shows also that the model scale is very important for the ability of the model to benefit from instruction tuning.

![Screenshot](FLAN_instructiontuning.PNG)

FLAN-T5 model comes with many variants based on the numbers of parameters:

- FLAN-T5 small (60M)
- FLAN-T5 base (250M)
- FLAN-T5 large (780M)
- FLAN-T5 XL (3B)
- FLAN-T5 XXL (11B)



- [Full Finetuning of base Flan T5](https://www.philschmid.de/fine-tune-flan-t5) : In this experiment we do full finetuning of an instruction tuned model (FLAN) without leveraging efficient techniques. The finetuning is a one single task (dialogue summarization) instruction tuning. To reproduce the same experiment I had to use the Trainer on a T4 free colab but with a gradient_checkpointing and a combination of train_batch_size=32, fp16 mixed precision (I didn't finish the training, but it is said that fp16 leads to overflow issues with T5). It took 4 hours of training duration for the T5 base model (which is 250M parameters large). I tried to run the same experiment on a colab pro with NVIDIA V100 but it didn't work. I suppose that the memory of the p3.2xlarge AWS EC2 Instance use in the tutorial is a little bit larger. [Colab notebook](https://colab.research.google.com/drive/1_RZgtC-_cZUCrInpsLQwmZrIY-ijvNFR?usp=sharing)
- [Full Finetuning of XL/XXL Flan T5 with deepspeed](https://www.philschmid.de/fine-tune-flan-t5-deepspeed) : In this experiment we do full finetuning of an instruction tuned model (FLAN) without leveraging efficient techniques. The finetuning is a one single task (news articles summarization) instruction tuning. The XXL FLAN T5 is an 11B model. there is no way to fully finetune the model in colab. the tutorial uses deepspeed on large EC2 instances with multiple GPUs.
It starts with the preprocessing outside of the GPU instance then loading the tokenized dataset from disk and the deepspeed config file in the train script like this :
![Screenshot](sdfrere.PNG)
Actually different experiments were conducted in this tutorial using different config deepspeed files:
![Screenshot](sdlkheio.PNG)
As fp16 can overflow, bf16 is the better choice because it provides significant advantages over fp32. We see also that it is better to keep a small batch size and not do offloading than the other way around. [Colab notebook](https://colab.research.google.com/drive/1Kl2ojG83-cWTip9-_hj_2mH7rTTBj5Pj?usp=sharing)
- [Finetune XXL Flan T5 with Lora and 8bit quant](https://www.philschmid.de/fine-tune-flan-t5-peft) : In this tutorial, the 11B FLAN-T5 XXL was finetuned using Lora and 8bit quantization(bnb) from Peft.
The same techniques were used for data preprocessing. The finetuning is a one single task (dialogue summarization) instruction tuning.
The only differences was the construction of the peft model and the preparing for int8 training:
![Screenshot](Peft_training.PNG)
This configuration uses only 0.16% of the parameters of the model. The training took 10h and cost ~13.22$. A full fine-tuning on FLAN-T5-XXL with the same duration (10h) requires 8x A100 40GBs and costs ~322$.
During evaluation, the results were slightly better than a full finetuning of FLAN-T5 base. [Colab notebook](https://colab.research.google.com/drive/1S5L1HvYv61oVKH9aZZ26nJXNt4akivN7?usp=sharing)

Another notebook was created for comparing full fine-tuning and Peft lora fine-tuning on the base FLAN-T5. [Colab notebook](https://colab.research.google.com/drive/18EzRa2oSfjOQBYz1SnCnmjLMn-X4S01h?usp=sharing)

-Bloom:

Training a 176 Billion parameter model needed the following hardware/software : 

![Screenshot](BLOOM.PNG)

- [Finetune 7B BLOOMZ with lora](https://www.philschmid.de/bloom-sagemaker-peft) : BLOOMZ 7b1 is the finetuned version from the pretrained [bloom 7b1](https://huggingface.co/bigscience/bloom-7b1). there is also a finetuned version for prompting in non english (ex: French) [bloomz-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt) a 7B parameters model, and a bigger one [bloomz-mt](https://huggingface.co/bigscience/bloomz-mt) of 176B parameters. The mt versions are finetuned on xP3mt, a Mixture of 13 training tasks in 46 languages with prompts in 20 languages (machine-translated from English).

  In this tutorial, the 7B BLOOMZ was finetuned using Lora and 8bit quantization(bnb) from Peft. The finetuning is a one single task (dialogue summarization) instruction tuning.

-LLAMA2:

- Checkout [Ressources about LLama2](https://www.philschmid.de/llama-2)

LLaMA 2 is a large language model developed by Meta and is the successor to LLaMA 1. LLaMA 2 pretrained models are trained on 2 trillion tokens, and have double the context length than LLaMA 1. Its fine-tuned models have been trained on over 1 million human annotations.

![Screenshot](LLAMA2.PNG)


Llama 2 outperforms other open source language models on many external benchmarks, including reasoning, coding, proficiency, and knowledge tests.

![Screenshot](LLAMA_eval.PNG)

However, the most exciting part of this release is the fine-tuned models (Llama 2-Chat), which have been optimized for dialogue applications using Reinforcement Learning from Human Feedback (RLHF).

- [Instruction tuning LLama2 with trl and SFTTrainer](https://www.philschmid.de/instruction-tune-llama-2)
  In this blog, it was used trl and SFTTrainer with Qlora to fine-tune Llama 7b version (not the chat version). To do so, bitsandbytes was used with some special Llama configurations to load the model in 4bit:
  ![Screenshot](Load_Llama.PNG)
  The lora adapters were prepared as usual:
  ![Screenshot](Lora_adapters.PNG)
  Finally, the trainer is prepared with trl:
  ![Screenshot](trl_trainer.PNG)

  Note that this way, we did not have to do the tokenization (with the truncation & padding) nor preparing the data collators beforehand. The SFTTrainer from trl took care of that. Checkout this [Colab notebook](https://colab.research.google.com/drive/1KgC3TUBIDBf-tsuLYKiAnUdE2-mOLzZg)
   
- [Finetune LLAMA2 using Qlora](https://www.philschmid.de/sagemaker-llama2-qlora)
  In this blog, LLaMA2 was instruction-tuned (on multi task instructions using dolly 15k dataset) using Qlora to fit training into a T4 tier colab. Comparing this experiment to the previous one, additional preprocessing was conducted to pack multiple samples to one sequence of 2048 length to have a more efficient training so that we passed from 15K samples (to train on in the previous experiment) to 1591 sequence to train on.

  As we did this additional preprocessing, we had to tokenize our input sequences and as result to use the simple Trainer and not the TRL SFTtrainer.

  Finally when comparing the two experiments (using 4 accumulation steps and a train batch size of 2) on a T4 free colab GPU: the first experiment would take 330h and the second would take 32h. If I choose a batch size larger than 2 I get CUDA out of memory. Checkout this [Colab notebook](https://colab.research.google.com/drive/1OOtPNwJLa3upPqGFkzJYxauo0YTSWtZ4)
    
- [Deploying LLama2 on Sagemaker](https://www.philschmid.de/sagemaker-llama-llm)

-Falcon:
- [Finetune Falcon with QLora](https://www.philschmid.de/sagemaker-falcon-qlora)
  I tried to follow this blog and instruct-tune (Multi task instructions using the Dolly15K dataset) a pretrained Falcon-7B model on T4 GPU free colab, although in the blog we finetune the 40B model on SagemMaker.

  Basically the blog uses the same data, preprocessing steps, and same parameters for the Lora adapters and the model, so we preprocess, tokenize and pack multiple samples to sequences of 2048 length. Using the Falcon tokenizer we get 1300 sample instead of 1500 sample with the LLaMA tokenizer on the same data.

  We use the simple Trainer for training (not SFTTrainer), we launch training but we hit Cuda out of memory, I think because I am using more parameters for Falcon (Falcon has 200M additional parameters than LLaMA for the 7B versions). Actually for this experiment I used a function copied from https://github.com/artidoro/qlora/blob/main/qlora.py to identify the layers for which we would create adapters. Creating adapters for those layers with 64 rank I am almost training 2% of the model parameters. so I tried to reduce the rank from 64 to 32 and 16 and tried to reduce the number of layers to which create adapters but still hit out of memory even with 0.1% lora parameters. So I conclude that it is not possible to train The 7B Falcon on the T4 GPU colab. Checkout this [Colab notebook](https://colab.research.google.com/drive/18ZCxs73eIzOqxVzoJvKMtlTif1apS6fW#scrollTo=WFI_msO_J5X-)
  
- [Finetune Falcon using Qlora with Flash Attention](https://www.philschmid.de/sagemaker-falcon-180b-qlora)
- [Finetune Falcon with deepspeed and Lora](https://www.philschmid.de/deepspeed-lora-flash-attention)

-Mistral:


-Compare GPTQ and bnb:
- [GPTQ and Hugging Face Optimum](https://www.philschmid.de/gptq-llama)
