# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

import torch.nn.functional as F
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_distill:
            self.voc_length = len(self.tokenizer.get_vocab())

        if finetuning_args.freq_tokens_path:
            self.freq_tokens = json.load(open(finetuning_args.freq_tokens_path))

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))

    def generate_teacher_logits(self, logit_tensor):

        # Step 1: 提取 token 和 logprobs
        tokens = logit_tensor[:, :, :, 0].to(torch.int)  # [batch_size, num_token, num_logprobs]
        logits = torch.exp(logit_tensor[:, :, :, 1])  # [batch_size, num_token, num_logprobs]

        # Step 2: 初始化一个形状为 [1, num_token, voc_length] 的稀疏向量，并填充 logprobs 值
        batch_size, num_token, num_logprobs = tokens.shape
        output_tensor = torch.zeros((batch_size, num_token, self.voc_length)).to(logit_tensor.device)  # [1, num_logprobs, voc_length]

        if self.finetuning_args.freq_tokens_path:
            for k in range(self.finetuning_args.num_freq_tokens):
                output_tensor[:, :, self.freq_tokens[k]['token_id']] = \
                    self.finetuning_args.distill_beta * self.freq_tokens[k]['prob']

        logits[:, 0] = 0

        for i in range(batch_size):
            for j in range(num_token):
                for k in range(num_logprobs):
                    if tokens[i, j, k] != -100:
                        output_tensor[i, j, tokens[i, j, k]] = self.finetuning_args.distill_gamma * logits[i, j, k] # 在对应的 token 位置填充 logit 值

        return F.normalize(output_tensor, p=1, dim=-1) # F.softmax(output_tensor, dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False):
        # assert inputs['input_ids'].size(0) == 1, 'current only support batch_size = 1'

        logprobs = inputs.pop('logprobs') if 'logprobs' in inputs else None

        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        student_logits = outputs_student.logits
        mask = (inputs['labels'] != -100)  # 创建一个布尔掩蔽张量，True 表示非 -100 的位置
        bs = mask.shape[0]

        if self.finetuning_args.use_distill and len(logprobs.shape)>3:

            teacher_softmax_logits = self.generate_teacher_logits(logprobs)

            # poll size
            # student_logits, teacher_softmax_logits = self.pool_to_same_tokens(student_logits, teacher_softmax_logits)
            assert student_logits.size() == teacher_softmax_logits.size(), f'{student_logits.size()} != {teacher_softmax_logits.size()}'

            # Soften probabilities and compute distillation loss
            loss_logits = 0
            if not (student_logits == teacher_softmax_logits).all():
                for i in range(bs):
                    masked_student_logits = F.softmax(student_logits[i][mask[i]][:-1], dim=-1)
                    masked_teacher_logits = teacher_softmax_logits[i][mask[i]][1:]
                    if self.finetuning_args.use_progressive_distill:
                        delta = self.state.epoch / self.state.num_train_epochs
                        masked_teacher_logits = masked_teacher_logits * delta + masked_student_logits * (1-delta)
                    loss_logits += F.kl_div(torch.log(masked_student_logits), masked_teacher_logits, reduction="batchmean")
                    # -torch.sum(torch.log(masked_student_logits)*masked_teacher_logits)/masked_student_logits.shape[0]
                    # F.cross_entropy(torch.log(masked_student_logits), masked_teacher_logits)
                    # F.kl_div(torch.log(masked_student_logits), masked_teacher_logits, reduction="batchmean")
                loss_logits /= bs

            # Return weighted student loss
            loss = self.finetuning_args.distill_alpha * student_loss + (1.0 - self.finetuning_args.distill_alpha) * loss_logits
            return (loss, outputs_student) if return_outputs else loss

        else:
            if len(logprobs.shape)>3:
                print(f'Warning! logprobs is with wrong shape {logprobs.shape}')
            loss = student_loss
            return (loss, outputs_student) if return_outputs else loss
