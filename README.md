# DiffusionRank: Effective Document Ranking with Diffusion Large Language Models

## Training

### Training Data

We preprocess the training data from [](https://huggingface.co/datasets/castorini/rank_zephyr_training_data), which is a collection of 40k query and ranked documents labeled by GPT-4 and used for the training of RankZephyr. The training data is preprocessed and converted into a format suitable for training DiffusionRank.

### Training Scripts

To train the model, you can use the following command:

```bash
python train.py configs/llada1.5_rmask_lora.yaml
```

## Evaluation

### Models

We release the trained DiffusionRank model, which can be used for evaluation on various document ranking benchmarks:

- [DiffuRank_Pointwise](https://huggingface.co/liuqi6777/DiffuRank_Pointwise): A pointwise ranking model that predicts the relevance score of each document independently.
- [DiffuRank_LogitsListwise](https://huggingface.co/liuqi6777/DiffuRank_LogitsListwise): A listwise ranking model that predicts the relevance scores of all documents in a list simultaneously.
- [DiffuRank_PermutationListwise](https://huggingface.co/liuqi6777/DiffuRank_PermutationListwise): A listwise ranking model that predicts the permutation of the documents in a list.

For more details about the models, please refer to the paper.

### Evaluation Scripts

To evaluate the models on the MS MARCO document ranking benchmark, you can use the following command:

```bash
# Evaluate the pointwise model
python src/eval.py \
  --model liuqi6777/DiffuRank_Pointwise \
  --rerank-method pointwise \
  --reranking-args truncate_length=256 \
  --datasets dl19 dl20 covid nfc touche dbpedia scifact signal news robust04 \
  --topk 100 \
  --output-dir results/eval.jsonl

# Evaluate the logits-based listwise model
python src/eval.py \
  --model liuqi6777/DiffuRank_LogitsListwise \
  --rerank-method logits_listwise \
  --reranking-args truncate_length=256,window_size=20,step=10 \
  --datasets dl19 dl20 covid nfc touche dbpedia scifact signal news robust04 \
  --topk 100 \
  --output-dir results/eval.jsonl

# Evaluate the permutation-based listwise model with sampling inference strategy
python src/eval.py \
  --model liuqi6777/DiffuRank_PermutationListwise \
  --rerank-method permutation_listwise \
  --reranking-args truncate_length=256,window_size=20,step=10 \
  --model-args num_samples=1,inference_strategy=sampling \
  --datasets dl19 dl20 covid nfc touche dbpedia scifact signal news robust04 \
  --topk 100 \
  --output-dir results/eval.jsonl

# Evaluate the permutation-based listwise model with assignment inference strategy
python src/eval.py \
  --model liuqi6777/DiffuRank_PermutationListwise \
  --rerank-method permutation_listwise \
  --reranking-args truncate_length=256,window_size=20,step=10 \
  --model-args num_samples=1,inference_strategy=assignment \
  --datasets dl19 dl20 covid nfc touche dbpedia scifact signal news robust04 \
  --topk 100 \
  --output-dir results/eval.jsonl
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{liu2026diffurankeffectivedocumentreranking,
      title={DiffuRank: Effective Document Reranking with Diffusion Language Models}, 
      author={Qi Liu and Kun Ai and Jiaxin Mao and Yanzhao Zhang and Mingxin Li and Dingkun Long and Pengjun Xie and Fengbin Zhu and Ji-Rong Wen},
      year={2026},
      eprint={2602.12528},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2602.12528}, 
}
```