import contextlib
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import DataLoader, RandomSampler, Sampler
from trl import SFTTrainer

from transformers import Seq2SeqTrainer, TrainingArguments
from transformers import logging as hf_logging
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


class PackingSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
    ):
        self.dataset = dataset

        self.packing_strategies = self._get_packing_strategies(
            lengths=lengths,
            max_seq_len=max_seq_len,
            max_seq_per_pack=max_seq_per_pack,
        )

        self.lengths = lengths
        self.packing_sample_ls = self._transform_length_to_indices(
            strategies_per_length=self.packing_strategies,
            lengths=self.lengths,
        )

    def _get_packing_strategies(
        self,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
    ) -> dict:
        def add_pack(
            pack: List[int],
            count: int,
            tmp: defaultdict,
            final: defaultdict,
            limit: int,
            offset: int,
        ) -> None:
            if len(pack) == limit or offset == 0:
                final[offset].append((count, pack))
            else:
                tmp[offset].append((count, pack))

        seq_lens, counts = np.unique(lengths, return_counts=True)
        histogram = np.zeros(max_seq_len, dtype=np.int64)
        histogram[seq_lens - 1] = counts

        reversed_histogram = np.flip(histogram)

        tmp_strategies_per_length = defaultdict(list)
        strategies_per_length = defaultdict(list)

        for i in range(max_seq_len):
            n_sequences_to_bin = reversed_histogram[i]
            length_to_bin = max_seq_len - i
            offset = i + 1  # largest possible offset
            while n_sequences_to_bin > 0:
                if (length_to_bin + offset) in tmp_strategies_per_length:
                    # extract shortest pack that will get modified
                    n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                    new_pack = pack + [length_to_bin]
                    count = min(n_sequences_to_pack, n_sequences_to_bin)
                    if n_sequences_to_pack > n_sequences_to_bin:
                        # old pack gets reduced
                        n_sequences_to_pack -= n_sequences_to_bin
                        tmp_strategies_per_length[length_to_bin + offset].append((n_sequences_to_pack, pack))
                        n_sequences_to_bin = 0
                    else:
                        n_sequences_to_bin -= n_sequences_to_pack
                    add_pack(
                        new_pack, count, tmp_strategies_per_length, strategies_per_length, max_seq_per_pack, offset
                    )
                    # clean up to speed up main key search
                    if not tmp_strategies_per_length[length_to_bin + offset]:
                        tmp_strategies_per_length.pop(length_to_bin + offset)
                else:
                    offset -= 1
                # Does not fit anywhere. Create new pack.
                if offset < 0:
                    add_pack(
                        [length_to_bin],
                        n_sequences_to_bin,
                        tmp_strategies_per_length,
                        strategies_per_length,
                        max_seq_per_pack,
                        i,
                    )
                    n_sequences_to_bin = 0
        # merge all strategies
        for key in tmp_strategies_per_length:
            strategies_per_length[key].extend(tmp_strategies_per_length[key])

        return strategies_per_length

    def _transform_length_to_indices(self, strategies_per_length: dict, lengths: List[int]) -> List[List[int]]:
        length_to_indices = {}
        length_array = np.array(lengths)
        unique_lengths = np.unique(length_array).tolist()

        for length in unique_lengths:
            dataset_idx_ls = np.where(length_array == length)[0].tolist()
            random.shuffle(dataset_idx_ls)
            length_to_indices[length] = dataset_idx_ls

        pack_strategies_ls = [
            pack
            for strategies in strategies_per_length.values()
            for strategies_num, pack_strategies in strategies
            for pack in ([pack_strategies] * strategies_num)
        ]

        packing_sample_ls = list()
        for pack_strategies in pack_strategies_ls:
            pack_size = len(pack_strategies)
            strategie_position = 0

            dataset_idx_ls = list()
            while strategie_position + 1 <= pack_size:
                length = pack_strategies[strategie_position]
                pack_length_ls = length_to_indices[length]
                dataset_idx_ls.append(pack_length_ls.pop())
                length_to_indices[length] = pack_length_ls
                strategie_position += 1

            packing_sample_ls.append(dataset_idx_ls)

        random.shuffle(packing_sample_ls)

        return packing_sample_ls

    def __iter__(self):
        packing_sample_ls = self._transform_length_to_indices(
            strategies_per_length=self.packing_strategies,
            lengths=self.lengths,
        )

        return iter(packing_sample_ls)

    def __len__(self):
        return len(self.packing_sample_ls)


class PackingTrainer(Seq2SeqTrainer, SFTTrainer):
    def __init__(
        self,
        args,
        **kwargs,
    ) -> None:
        setattr(args, "dataset_kwargs", {"skip_prepare_dataset": True})
        setattr(args, "padding_free", False)
        SFTTrainer.__init__(
            self,
            args=args,
            **kwargs,
        )
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        def __packing_getitems__(train_dataset, keys: List[List[int]]) -> List:
            """Can be used to get a batch using a list of integers indices."""

            return_ls = list()
            for key in keys:
                batch = train_dataset.__getitem__(key)
                n_examples = len(batch[next(iter(batch))])

                return_ls.append([{col: array[i] for col, array in batch.items()} for i in range(n_examples)])
            return return_ls

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # NOTE: packing을 사용할 경우 packing에 알맞은 getitems를 사용하도록 합니다.
        if self.args.do_packing:
            # 래핑된 함수를 정의하여 self를 전달할 수 있도록 합니다.
            def getitems_wrapper(keys):
                return __packing_getitems__(train_dataset, keys)

            setattr(self.train_dataset, "__getitems__", getitems_wrapper)

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length and self.args.do_packing:
            raise ValueError("group_by_length and do_packing cannot be used together.")

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.args.do_packing:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None

            return PackingSampler(
                dataset=self.train_dataset,
                lengths=lengths,
                max_seq_len=self.args.data_max_length,
                max_seq_per_pack=self.args.packing_max_elem,
            )

        else:
            return RandomSampler(self.train_dataset)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super(Seq2SeqTrainer, self).prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        use_labels_at_compute_loss = False
        compute_loss = "labels" in inputs and use_labels_at_compute_loss
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
            and use_labels_at_compute_loss
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if compute_loss:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if compute_loss and self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels


class PackingCollatorForLLM(DataCollatorMixin):
    def __init__(
        self,
        args: TrainingArguments,
        model,
        tokenizer,
        sample_dataset: Optional[Dataset] = None,
        return_tensors: Optional[str] = "pt",
    ) -> None:
        self.args = args
        self.model = model
        self.return_tensors = return_tensors
        self.tokenizer = tokenizer

        sample_check = self([sample_dataset[0]])
        input_ids = sample_check["input_ids"].tolist()[0]
        if self.args.distributed_state.is_local_main_process:
            sample_check["labels"] = sample_check["labels"][sample_check["labels"] != -100].tolist()
            check_labels = [self.tokenizer.convert_ids_to_tokens(token) for token in sample_check["labels"]]
            check_labels = ", ".join(check_labels)
            logger.info(f"collator_label: [-100,  ..., -100, {check_labels}]")

        if self.tokenizer.bos_token_id and self.tokenizer.bos_token_id not in input_ids:
            raise ValueError("BOS token이 없다. 이거 다시 전처리 해라.")

        if self.tokenizer.eos_token_id not in input_ids:
            raise ValueError("EOS token이 없다. 이거 다시 전처리 해라.")

    def _create_attention_mask(self, input_length_ls: List[int]) -> torch.Tensor:
        total_length = sum(input_length_ls)
        attention_mask = torch.full((1, 1, total_length, total_length), torch.finfo(self.dtype).min)

        start_idx, end_idx = 0, 0
        for length in input_length_ls:
            end_idx += length
            one_tensor = torch.ones((length, length), dtype=torch.float32)
            mask = torch.tril(one_tensor, diagonal=0).to(dtype=torch.bool)
            attention_mask[0, 0, start_idx:end_idx, start_idx:end_idx][mask] = 0
            start_idx = end_idx

        return attention_mask

    def torch_call(
        self,
        features_ls: List[dict],
    ) -> dict:
        input_ids_ls, labels_ls, position_ids_ls, input_length_ls = list(), list(), list(), list()
        for packing_ls in features_ls:
            packing_ls = [packing_ls] if isinstance(packing_ls, dict) else packing_ls
            for features in packing_ls:
                if self.model.training:
                    length = len(features["input_ids"])

                    # input_ids_ls.append({"input_ids": features["input_ids"]})
                    # labels_ls.append({"input_ids": features["labels"]})
                    input_ids_ls.append(features["input_ids"])
                    labels_ls.append(features["labels"])
                    position_ids_ls.append(torch.arange(length))
                    input_length_ls.append(length)
                else:
                    input_ids_ls.append({"input_ids": features["input_ids"]})
                    labels_ls.append({"input_ids": features["labels"]})

        batch = dict()
        if input_ids_ls and self.model.training:
            # output = self.tokenizer.pad(input_ids_ls, padding_side="left", return_tensors="pt")
            # batch["input_ids"] = output.input_ids
            # batch["attention_mask"] = output.attention_mask
            batch["input_ids"] = torch.concat(input_ids_ls)[None]
        else:
            output = self.tokenizer.pad(input_ids_ls, padding_side="left", return_tensors="pt")
            batch["input_ids"] = output.input_ids
            batch["attention_mask"] = output.attention_mask

        if labels_ls and self.model.training:
            # output = self.tokenizer.pad(labels_ls, padding_side="left", return_tensors="pt")
            # for idx, (input_ids, attention_mask) in enumerate(zip(output.input_ids, output.attention_mask)):
            #     input_ids[attention_mask == 0] = -100
            #     output.input_ids[idx] = input_ids
            # batch["labels"] = output.input_ids
            batch["labels"] = torch.concat(labels_ls)[None]
        else:
            output = self.tokenizer.pad(labels_ls, padding_side="left", return_tensors="pt")
            batch["labels"] = output.input_ids

        if position_ids_ls and self.model.training:
            batch["position_ids"] = torch.concat(position_ids_ls)[None]

        if self.args.attn_implementation == "eager" and self.model.training:
            batch["attention_mask"] = self._create_attention_mask(input_length_ls)

        return batch
