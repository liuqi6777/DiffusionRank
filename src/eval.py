import argparse
import json
import os
import time
import random
from functools import partial
from transformers.utils import logging

from llm4ranking.evaluation.evaluator import evaluate
from llm4ranking.ranker.base import TournamentReranker, PointwiseReranker, ListwiseSilidingWindowReranker

from eval_utils import (
    LladaForEval,
    ListwiseGenerationWrapper,
    LogitsListwiseWrapper,
    PermutationListwiseWrapper,
    PointwiseWrapper,
)

logging.set_verbosity_error()


def parse_dict_args(args_string: str):
    args = {}
    for arg in args_string.split(","):
        key, value = arg.strip().split("=")
        try:
            args[key] = eval(value)
        except Exception:
            args[key] = value
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0)

    parser.add_argument("--rerank-method", type=str, default="permutation_listwise", choices=[
        "generation_listwise",
        "logits_listwise",
        "permutation_listwise",
        "pointwise"
    ])
    parser.add_argument("--reranking-args", type=parse_dict_args, default={})
    parser.add_argument("--model-args", type=parse_dict_args, default={})

    parser.add_argument("--gen-length", type=int, default=256, help="Response length")
    parser.add_argument("--steps", type=int, default=128, help="Diffusion steps")
    parser.add_argument("--block-size", type=int, default=256, help="Block size")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold value")
    parser.add_argument("--use-cache", help='Use cache', action='store_true')
    parser.add_argument("--dual-cache", help='Dual cache', action='store_true')
    parser.add_argument("--remasking", type=str, default='low_confidence', choices=['low_confidence', 'random'], help="Remasking strategy")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    print(args)

    random.seed(42)

    model = LladaForEval(
        model_path=args.model,
        rope_scaling_factor=args.rope_scaling_factor,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_size,
        remasking=args.remasking,
        threshold=args.threshold,
        use_cache=args.use_cache,
        dual_cache=args.dual_cache,
        device="cuda",
    )

    if args.rerank_method == "permutation_listwise_":
        ranker = ListwiseSilidingWindowReranker()
        rerank = partial(
            ranker.rerank,
            ranking_func=PermutationListwiseWrapper(model, **args.model_args),
            **args.reranking_args,
        )
    elif args.rerank_method == "generation_listwise":
        ranker = ListwiseSilidingWindowReranker()
        rerank = partial(
            ranker.rerank,
            ranking_func=ListwiseGenerationWrapper(model, **args.model_args),
            **args.reranking_args,
        )
    elif args.rerank_method == "logits_listwise":
        ranker = ListwiseSilidingWindowReranker()
        rerank = partial(
            ranker.rerank,
            ranking_func=LogitsListwiseWrapper(model, **args.model_args),
            **args.reranking_args,
        )
    elif args.rerank_method == "pointwise":
        ranker = PointwiseReranker()
        rerank = partial(
            ranker.rerank,
            **args.reranking_args,
            ranking_func=PointwiseWrapper(model, **args.model_args),
        )
    else:
        raise ValueError(f"Unknown rerank method: {args.rerank_method}")

    results = evaluate(
        rerank,
        datasets=args.datasets,
        retriever=args.retriever,
        topk=args.topk,
        output_dir=os.path.join("outputs", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S")),
    )

    if args.output_dir is not None:
        with open(args.output_dir, "a") as f:
            f.write(json.dumps(
                {"args": vars(args), "results": results},
                default=str,
            ) + "\n")
        print(f"Results saved to {args.output_dir}")

