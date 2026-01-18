import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from model.modeling_llada import LLaDAModelLM
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache


class LladaForEval:
    def __init__(
        self,
        model_path: str,
        rope_scaling_factor: float = 1.0,
        mask_id=126336,
        eos_id=126081,
        max_length=4096,
        batch_size=8,
        mc_num=16,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        threshold=None,
        use_cache=False,
        dual_cache=False,
    ):
        config = AutoConfig.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
        config.rope_theta = config.rope_theta * rope_scaling_factor
        self.model = LLaDAModelLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        self.model.eval()

        self.device = torch.device(device)
        self.model = self.model.to(device)

        self.mask_id = mask_id
        self.eos_id = eos_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.dual_cache = dual_cache

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        if self.dual_cache:
            output_ids, nfe, history = generate_with_dual_cache(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                remasking=self.remasking,
                threshold=self.threshold,
                mask_id=self.mask_id,
                eos_id=self.eos_id,
            )
        elif self.use_cache:
            output_ids, nfe, history = generate_with_prefix_cache(
                self.model,
                input_ids,
                steps=self.steps,
                max_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                remasking=self.remasking,
                threshold=self.threshold,
                mask_id=self.mask_id,
                eos_id=self.eos_id,
            )
        else:
            output_ids, nfe, history = generate(
                self.model,
                input_ids,
                steps=self.steps,
                max_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                remasking=self.remasking,
                threshold=self.threshold,
                mask_id=self.mask_id,
                eos_id=self.eos_id,
            )
        return output_ids, nfe, history

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index=None, cfg=0.):
        if cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)


class WrapperBase:
    def __init__(self, model: LladaForEval, **kwargs):
        self.model = model

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ListwiseGenerationWrapper(WrapperBase):

    _prompt_template = """Given a query and {num} documents indicated by a character identifier, rank the documents from most relevant to least relevant to the query.

You should output a ranking using the document identifier, from most relevant to least relevant, separated by spaces.

Query: {query}

Documents:
{documents}
"""

    def __call__(self, query: str, docs: list[str], **kwargs) -> list[int]:

        messages = [{"role": "user", "content": self._prompt_template.format(
            num=len(docs),
            query=query,
            documents="\n".join([f"Document {chr(ord('A') + i)}: {doc}" for i, doc in enumerate(docs)])
        )}]
        input_ids = self.model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids, _, history = self.model.generate(input_ids)
        output_text = self.model.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

        def parse_output(output: str, n: int) -> list[int]:
            permutation = []
            for ch in re.findall(r"[A-Za-z]", output):
                idx = ord(ch.upper()) - ord('A')
                if 0 <= idx < n:
                    permutation.append(idx)
            permutation = list(dict.fromkeys(permutation))  # remove duplicates
            permutation = [x for x in permutation if x in range(n)] + [x for x in range(n) if x not in permutation]
            return permutation

        ranking = parse_output(output_text, len(docs))
        outputs = {
            "inputs": {
                "query": query,
                "documents": docs,
                "prompts": self.model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            },
            "output": {
                "output_text": output_text,
                "history": [self.model.tokenizer.batch_decode(x[:, input_ids.shape[1]:], skip_special_tokens=False)[0] for x in history],
            }
        }
        return ranking, outputs


class PointwiseWrapper(WrapperBase):
    _prompt_template = """Given a query and a document, determine whether the document is relevant to the query. The relevance score should be either 0 (not relevant) or 1 (relevant).

Query: {query}

Document:
{doc}"""

    def __call__(self, query: str, doc: str, **kwargs):
        prompt = self._prompt_template.format(query=query, doc=doc)
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "<|mdm_mask|>"}]
        input_ids = self.model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(self.model.device)
        yes_loc = self.model.tokenizer.encode("1")[0]
        no_loc = self.model.tokenizer.encode("0")[0]
        logits = self.model.get_logits(input_ids)
        mask_batch_idx, mask_token_idx = (input_ids == self.model.mask_id).nonzero(as_tuple=True)
        masked_logits = logits[mask_batch_idx, mask_token_idx, :]
        probs = torch.softmax(masked_logits, dim=-1)
        p_yes = probs[0, yes_loc].item()
        p_no = probs[0, no_loc].item()
        score = p_yes / (p_yes + p_no)
        outputs = {
            "output": {
                "probabilities": [p_yes, p_no],
                "score": score,
            }
        }
        return score, outputs


class LogitsListwiseWrapper(WrapperBase):
    _prompt_template = """Given a query and {num} documents indicated by a numeric identifier, determine whether each document is relevant to the query. The relevance score should be either 0 (not relevant) or 1 (relevant).

Query: {query}

Documents:
{documents}
"""

    def __call__(self, query: str, docs: list[str], **kwargs) -> list[float]:
        messages = [
            {
                "role": "user", 
                "content": self._prompt_template.format(
                    num=len(docs),
                    query=query,
                    documents="\n".join([f"Document {i}: {doc}" for i, doc in enumerate(docs)]),
                )
            },
            {"role": "assistant", "content": "\n".join([f"Document {i}: <|mdm_mask|>" for i in range(len(docs))])}
        ]
        input_ids = self.model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(self.model.device)
        yes_loc = self.model.tokenizer.encode(f"1")[0]
        no_loc = self.model.tokenizer.encode(f"0")[0]

        mask_batch_idx, mask_token_idx = (input_ids == self.model.mask_id).nonzero(as_tuple=True)
        logits = self.model.get_logits(input_ids)  # [B, T, V]
        masked_logits = logits[mask_batch_idx, mask_token_idx, :]

        probs = torch.softmax(masked_logits, dim=-1)  # [num_masks, V]
        p_yes = probs[:, yes_loc]
        p_no = probs[:, no_loc]
        scores = (p_yes / (p_yes + p_no)).tolist()
        ranking = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        outputs = {
            "inputs": {
                "query": query,
                "documents": docs,
                "prompts": self.model.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            },
            "output": {
                "probabilities": [[p_yes_0, p_no_0] for p_yes_0, p_no_0 in zip(p_yes.tolist(), p_no.tolist())],
                "scores": scores,
            }
        }
        return ranking, outputs



class PermutationListwiseWrapper(WrapperBase):
    _prompt_template = """Given a query and {num} documents indicated by a character identifier, rank the documents from most relevant to least relevant to the query.

You should output a ranking using the document identifier, from most relevant to least relevant, separated by spaces.

Query: {query}

Documents:
{documents}
"""

    def __init__(
        self,
        model,
        num_steps: int = 1,
        inference_strategy: str = "assignment",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.num_steps = int(num_steps)
        assert inference_strategy in ["assignment", "sampling"]
        self.inference_strategy = inference_strategy

    def __call__(
        self,
        query: str,
        docs: list[str],
        **kwargs,
    ):

        n = len(docs)
        if n <= 0:
            return [], {"inputs": {"query": query, "documents": docs}, "output": {"final_ranking": [], "steps": []}}

        mask_tok = "<|mdm_mask|>"

        index_strs = [f" {chr(i + ord('A'))}" for i in range(n)]
        index_token_ids = self.model.tokenizer.encode("".join(index_strs), add_special_tokens=False)
        id_to_idx = {s: i for i, s in enumerate(index_strs)}

        docs_block = "\n".join(
            [f"Document {chr(i + ord('A'))}: {doc}" for i, doc in enumerate(docs)]
        )

        ranking_tokens_str = [mask_tok] * n

        all_steps_info = []

        for step in range(self.num_steps):
            frac = float(step + 1) / float(self.num_steps)
            target_filled = min(n, max(1, int(round(frac * n))))

            current_filled_positions = [
                i for i, tok in enumerate(ranking_tokens_str)
                if tok != mask_tok
            ]
            current_filled = len(current_filled_positions)

            num_to_new_fill = max(0, target_filled - current_filled)

            ranking_line = (
                "Ranking (most to least relevant):" + "".join(ranking_tokens_str)
            )

            messages = [
                {"role": "user", "content": self._prompt_template.format(num=n, query=query, documents=docs_block)},
                {"role": "assistant", "content": ranking_line},
            ]

            input_ids = self.model.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(self.model.device)

            mask_batch_idx, mask_token_idx = (input_ids == self.model.mask_id).nonzero(as_tuple=True)

            if mask_token_idx.numel() == 0:
                break

            logits = self.model.get_logits(input_ids)
            masked_logits = logits[mask_batch_idx, mask_token_idx, :]
            probs = torch.softmax(masked_logits, dim=-1)

            S_partial = probs[:, index_token_ids]  # [M, n]
            S_partial_np = S_partial.float().detach().cpu().numpy()

            used_docs = set()
            for tok in ranking_tokens_str:
                if tok != mask_tok and tok in id_to_idx:
                    used_docs.add(id_to_idx[tok])

            mask_positions = [
                i for i, tok in enumerate(ranking_tokens_str)
                if tok == mask_tok
            ]
            M = len(mask_positions)
            assert M == S_partial_np.shape[0], "mask 数量和 S_partial 行数不一致"
            avail_docs = [j for j in range(n) if j not in used_docs]
            A = len(avail_docs)

            new_fills_this_step = []  # (rank_pos, doc_j, prob)

            if num_to_new_fill > 0 and A > 0 and M > 0:
                S_avail = S_partial_np[:, avail_docs]  # [M, A]

                if self.inference_strategy == "assignment" and S_avail.size > 0:
                    # Hungarian：min cost = -log(p)
                    cost = -np.log(np.clip(S_avail, 1e-12, 1.0))
                    row_ind, col_ind = linear_sum_assignment(cost)
                    assign_probs = S_avail[row_ind, col_ind]

                    order = np.argsort(-assign_probs)
                    k = min(num_to_new_fill, len(order))

                    for t in order[:k]:
                        m = int(row_ind[t])
                        a = int(col_ind[t])
                        doc_j = int(avail_docs[a])
                        prob_ = float(assign_probs[t])

                        rank_pos = mask_positions[m]
                        ranking_tokens_str[rank_pos] = index_strs[doc_j]
                        pos_conf[rank_pos] = prob_
                        used_docs.add(doc_j)
                        new_fills_this_step.append((rank_pos, doc_j, prob_))

                else:
                    # Greedy sampling fallback
                    pair_list = []
                    for m in range(M):
                        for a, doc_j in enumerate(avail_docs):
                            pair_list.append((S_avail[m, a], m, doc_j))
                    pair_list.sort(key=lambda x: x[0], reverse=True)

                    assigned_m = set()
                    newly_used = set()
                    filled = 0
                    for score, m, doc_j in pair_list:
                        if filled >= num_to_new_fill:
                            break
                        if m in assigned_m or doc_j in newly_used or doc_j in used_docs:
                            continue
                        assigned_m.add(m)
                        newly_used.add(doc_j)

                        rank_pos = mask_positions[m]
                        ranking_tokens_str[rank_pos] = index_strs[doc_j]
                        pos_conf[rank_pos] = float(score)
                        used_docs.add(doc_j)
                        new_fills_this_step.append((rank_pos, doc_j, float(score)))
                        filled += 1

            all_steps_info.append(
                {
                    "step": step,
                    "ranking_tokens_str": ranking_tokens_str.copy(),
                    "target_filled": target_filled,
                    "num_to_new_fill": num_to_new_fill,
                    "new_fills_this_step": new_fills_this_step,
                    "used_docs_after_step": sorted(list(used_docs)),
                    "assignment": "hungarian" if _HAS_SCIPY else "greedy_fallback",
                }
            )

            if all(tok != mask_tok for tok in ranking_tokens_str):
                break

        final_ranking = []
        for tok in ranking_tokens_str:
            if tok in id_to_idx:
                final_ranking.append(id_to_idx[tok])
            else:
                used = set(final_ranking)
                candidates = [i for i in range(n) if i not in used]
                final_ranking.append(candidates[0] if candidates else 0)

        seen = set()
        cleaned = []
        for idx in final_ranking:
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                cleaned.append(idx)
        for i in range(n):
            if i not in seen:
                cleaned.append(i)
        final_ranking = cleaned

        outputs = {
            "inputs": {
                "query": query,
                "documents": docs,
                "prompts": self.model.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                ),
                "num_steps": self.num_steps,
            },
            "output": {
                "final_ranking": final_ranking,
                "final_ranking_identifiers": [index_strs[i] for i in final_ranking],
                "steps": all_steps_info,
            },
        }

        return final_ranking, outputs