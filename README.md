# Vietnamese-LLM-7B

<div align="center">
    <img src="https://github.com/hahuyhoang411/Vietnamese-LLM-7B/assets/64120343/c8260f79-1a95-46ac-8194-215c8a56af63" width="300" height="300">
</div>


<p align="center">
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/hahuyhoang411/Vietnamese-LLM-7B"/>
  <img alt="Github Last Commit" src="https://img.shields.io/github/last-commit/hahuyhoang411/Vietnamese-LLM-7B"/>
  <img alt="Github Contributors" src="https://img.shields.io/github/contributors/hahuyhoang411/Vietnamese-LLM-7B"/>
  <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/hahuyhoang411/Vietnamese-LLM-7B"/>
</p>


# Introduction
This repository is for the VietAI Research team to expand the vocabulary of Open Source Large Language Models.

# Objective
- [ ] 1. Extend vocabulary for the base model (Llama-2/ Mistal).
    - [ ] Train a tokenizer for Vietnamese.
    - [ ] Merge vocabulary.
- [ ] 2. Continue training the `Llama2-7b` / `Mistral-7b` models on 120GB of Vietnamese data from the `CulturaX` dataset.
- [ ] 3. Perform supervised fine-tuning of the model using the `ultrachat-200k` dataset from `H4`.
- [ ] 4. Conduct DPO training of the model with the `ultrafeedback` / `no robots` datasets from H4.
- [ ] 5. Train for tool use. (Optional)

# Reference

## Extend vocab:
- [Chinese llama](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [Medical GPT](https://github.com/shibing624/MedicalGPT/blob/main/build_domain_tokenizer.py)

## SFT + DPO
- [Zephyr handbook](https://github.com/huggingface/alignment-handbook)
- [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)
- [Deepspeed Chat](https://medium.com/@musicalchemist/rlhf-training-at-scale-with-deepspeed-chat-6259bc04dc59)

## LORA
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

## Models
- [llama 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)

## Dataset
- [Pretrain](https://huggingface.co/datasets/uonlp/CulturaX)
- [SFT-chat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [DPO-ultrafb](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [DPO-norobot](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

