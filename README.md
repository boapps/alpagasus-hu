<h1 align="center">AlpaGasus: Training a Better Alpaca Model with Fewer Data</h1>
The unofficial implementation of "AlpaGasus: Training a better Alpaca with Fewer data." Trained models are available at the Huggingface and we will keep updating the filtered data.

## [Project page](https://lichang-chen.github.io/AlpaGasus/) | [Paper](https://arxiv.org/abs/2307.08701) | [Huggingface](https://huggingface.co/gpt4life/)

This repo contains:

- The filtered data used for instruction-finetuning the model.
- The code for filtering the data.
- The scripts for fine-tuning the model.
- The code and scripts for the evaluations. 

Note: thanks to the community for providing useful feedbacks, which really stimulates us to a better open-source.

<p align="center">
    <img src="alpagasus.jpeg" width="30%"> <br>
    Our Model "AlpaGasus"is pronounced as "/ˈælpəˈɡeɪsəs/", or "/ˈælpəˈɡəsəs/". The logo is generated by <a href="https://www.midjourney.com/app/">Midjourney</a>
</p>


## News
- [2023.9] We really appreciate the effort by [@gauss5930](https://github.com/gauss5930) who implemented the QLoRA version of Alpagasus-7B and 13B. If you do not have enough computational resources, please refer to their repo: [Alpagasus2-QLoRA](https://github.com/gauss5930/AlpaGasus2-QLoRA)
- [2023.9] We will release more filtered datasets! Stay tuned! 


## Setup
- Set up the environment of [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
- Stay tuned!

## Rating
Rate each (instruction, input, output) tuple in the Alpaca's 52k training set.
```
# Use ChatGPT as the response quality evaluator
export YOUR_OPENAI_API_KEY
# Use Claude as the response quality evaluator
export YOUR_CLAUDE_API_KEY
```
After the rating, you will need to use `rating/filter.py` and `rating/get_scores.py` to process your reviews obtained from ChatGPT/Claude.

## The structure of data

We provide the filtered data here for reproducing the results in our paper: `data/filtered/claude_t45.json` and `data/filtered/chatgpt_9k.json`. t45 means the threshold is 4.5 and chatgpt/claude means the prompted models. We also provide with the `data/random/random_9k.json` file which is randomly selected from the original Alpaca dataset.


## Training
- For the instruction-finetuning of LLaMA-7B: 
```
# prepare the data 
sh training/train_7b.sh
```
- For the instruction-finetuning of LLaMA-13B:
```
sh training/train_13b.sh
```



## Other tests
We use ChatGPT as the grader to evaluate the model's output.
```
export OPENAI_API_KEY
cd evaluation/
sh run_eval.sh
```


## References
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [Koala](https://github.com/young-geng/EasyLM/tree/main)
- [Vicuna](https://vicuna.lmsys.org/)
- [GPT-4-Report](https://arxiv.org/pdf/2303.08774.pdf)

## Citation
If you think it is a useful repo, please cite the paper:
```bibtex
@misc{chen2023alpagasus,
      title={AlpaGasus: Training A Better Alpaca with Fewer Data}, 
      author={Lichang Chen and Shiyang Li and Jun Yan and Hai Wang and Kalpa Gunaratna and Vikas Yadav and Zheng Tang and Vijay Srinivasan and Tianyi Zhou and Heng Huang and Hongxia Jin},
      year={2023},
      eprint={2307.08701},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
