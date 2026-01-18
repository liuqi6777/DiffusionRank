import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer


def forward_process(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    mask_token_id: int = 126336,
    eps: float = 1e-3
):
    B, L = input_ids.shape
    device = input_ids.device

    t = torch.rand(B, device=device)
    p_mask_vals = (1 - eps) * t + eps
    rand_prob = torch.rand(B, L, device=device)
    mask_cond = rand_prob < p_mask_vals.unsqueeze(1)

    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    resp_mask = pos >= prompt_lengths.unsqueeze(1)
    mask_cond &= resp_mask

    noisy_batch = torch.where(mask_cond, mask_token_id, input_ids)
    p_mask_batch = torch.where(mask_cond, p_mask_vals.unsqueeze(1), 1.0)
    mask_idx_batch = mask_cond

    return noisy_batch, mask_idx_batch, p_mask_batch


def ranking_aware_forward_process(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    step: int,
    total_steps: int,
    docid_token_ids: list[int],
    mask_token_id: int = 126336,
    mask_ratio_range: tuple[float] = (0., 1.),
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, L = input_ids.shape
    device = input_ids.device

    t = torch.rand(B, device=device)
    p_mask_vals = (1 - eps) * t + eps
    rand_prob = torch.rand(B, L, device=device)
    mask_cond = rand_prob < p_mask_vals.unsqueeze(1)

    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    resp_mask = pos >= prompt_lengths.unsqueeze(1)
    mask_cond &= resp_mask

    docid_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    for docid_token_id in docid_token_ids:
        docid_mask |= (input_ids == docid_token_id)
    docid_mask &= resp_mask

    mask_cond &= docid_mask

    noisy_batch = torch.where(mask_cond, mask_token_id, input_ids)
    p_mask_batch = torch.where(mask_cond, p_mask_vals.unsqueeze(1), 1.0)
    mask_idx_batch = mask_cond

    return noisy_batch, mask_idx_batch, p_mask_batch


class SFTTrainer(Trainer):
    
    def __init__(self, mask_token_id=126336, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.docid_token_ids = self.tokenizer.encode(" A B C D E F G H I J K L M N O P Q R S T U V W X Y Z")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]

        if self.args.mask_strategy == 'default':
            noisy_batch, masked_indices, p_mask = forward_process(
                input_ids,
                prompt_lengths,
                mask_token_id=self.mask_token_id,
            )
        elif self.args.mask_strategy == 'ranking_aware':
            noisy_batch, masked_indices, p_mask = ranking_aware_forward_process(
                input_ids,
                prompt_lengths,
                step=self.state.global_step,
                total_steps=self.args.max_steps,
                docid_token_ids=self.docid_token_ids,
                mask_token_id=self.mask_token_id,
                mode_probs=tuple(self.args.mode_probs),
                mask_ratio_range=tuple(self.args.mask_ratio_range),
                swap_ratio_range=tuple(self.args.swap_ratio_range),
                include_swap_in_loss=self.args.include_swap_in_loss,
            )
        else:
            raise ValueError(f"Unknown mask_strategy: {self.args.mask_strategy}")

        # Calculate the answer length (including the padded <EOS> tokens)
        answer_lengths = (input_ids.shape[1] - prompt_lengths).unsqueeze(1)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

        outputs = model(input_ids=noisy_batch)
        logits = outputs.logits
            
        token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
        ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]

        return (ce_loss, outputs) if return_outputs else ce_loss


class MultiDocLogitsTrainer(Trainer):
    
    def __init__(self, mask_token_id=126336, use_ranknet_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.yes_loc = self.tokenizer.encode(f"1")[0]
        self.no_loc = self.tokenizer.encode(f"0")[0]
        self.use_ranknet_loss = use_ranknet_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        ranking = inputs["ranking"]
        batch_size, slate_length = ranking.shape

        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        mask_indices = input_ids == self.mask_token_id
        assert mask_indices.sum(dim=1).eq(torch.tensor(len(ranking[0]), device=mask_indices.device)).all(), "Number of masked positions must equal number of documents"

        logits_at_mask = logits[mask_indices]  # (B * num_docs, vocab_size)

        if self.use_ranknet_loss:
            scores = logits_at_mask[:, self.yes_loc].view(batch_size, slate_length)
            rank_position = torch.empty_like(ranking, device=ranking.device, dtype=torch.long)
            rank_indices = torch.arange(slate_length, device=ranking.device).expand(batch_size, -1)
            rank_position.scatter_(dim=1, index=ranking, src=rank_indices)

            loss = rank_net(scores, slate_length - rank_position)  # use -rank_position as relevance

        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            scores = torch.softmax(logits_at_mask[:, [self.yes_loc, self.no_loc]], dim=-1)[:, 0].view(batch_size, slate_length)
            scores = scores / 0.1 # scale factor to make training more stable
            targets = ranking[:, 0]
            loss = loss_fn(scores, targets)

        return (loss, outputs) if return_outputs else loss


class PointwiseTrainer(Trainer):
    
    def __init__(self, mask_token_id=126336, use_ranknet_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.yes_loc = self.tokenizer.encode(f"1")[0]
        self.no_loc = self.tokenizer.encode(f"0")[0]
        self.use_ranknet_loss = use_ranknet_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        ranking = inputs["ranking"]
        batch_size, slate_length = ranking.shape

        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        mask_indices = input_ids == self.mask_token_id

        logits_at_mask = logits[mask_indices]  # (B, num_docs * vocab_size)
        

        if self.use_ranknet_loss:
            scores = logits_at_mask[:, self.yes_loc].view(batch_size, slate_length)
            rank_position = torch.empty_like(ranking, device=ranking.device, dtype=torch.long)
            rank_indices = torch.arange(slate_length, device=ranking.device).expand(batch_size, -1)
            rank_position.scatter_(dim=1, index=ranking, src=rank_indices)

            loss = rank_net(scores, slate_length - rank_position)  # use -rank_position as relevance

        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

            p_yes = torch.exp(logits_at_mask[:, self.yes_loc])
            p_no = torch.exp(logits_at_mask[:, self.no_loc])
            scores = (p_yes / (p_yes + p_no)).view(batch_size, slate_length)
            scores = scores / 0.1 # scale factor to make training more stable
            targets = ranking[:, 0]
            loss = loss_fn(scores, targets)

        return (loss, outputs) if return_outputs else loss


def rank_net(y_pred, y_true, weighted=False, use_rank=False, weight_by_diff=False, weight_by_diff_powed=False):
    if use_rank is None:
        y_true = torch.tensor([[1 / (np.argsort(y_true)[::-1][i] + 1) for i in range(y_pred.size(1))]] * y_pred.size(0)).cuda()

    document_pairs_candidates = list(itertools.product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weighted:
        values, indices = torch.sort(y_true, descending=True)
        ranks = torch.zeros_like(indices)
        ranks.scatter_(1, indices, torch.arange(1, y_true.numel() + 1).to(y_true.device).view_as(indices))
        pairs_ranks = ranks[:, document_pairs_candidates] 
        rank_sum = pairs_ranks.sum(-1)
        weight = 1 / rank_sum[the_mask]    
    else:
        if weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)
