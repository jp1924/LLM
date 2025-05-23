import numpy as np
import wandb
from accelerate import Accelerator
from evaluate import load
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from transformers import EvalPrediction, PreTrainedTokenizer, TrainingArguments


def compute_tnt_metrics(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizer,
    args: TrainingArguments,
) -> dict[str, float]:
    eval_pred.predictions[eval_pred.predictions == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=False)
    src = [x[0].replace(tokenizer.pad_token, "").split("<|im_start|>")[-1].split("<|im_end|>")[0] for x in predictions]

    eval_pred.label_ids[eval_pred.label_ids == -100] = tokenizer.pad_token_id
    label_ids = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=False)
    lbl = [x[0].replace(tokenizer.pad_token, "") for x in label_ids]

    wer_score_ls, cer_score_ls, result_ls = [], [], []
    wer, cer = load("evaluate-metric/wer"), load("evaluate-metric/cer")

    if "wandb" in args.report_to:
        predictions_table = wandb.Table(columns=["Prediction", "Label", "WER", "CER"])

        sample_ls = list(zip(lbl, src))
        data_loader = DataLoader(
            sample_ls,
            batch_size=1,
            sampler=DistributedSampler(
                sample_ls,
                num_replicas=args.world_size,
                rank=args.local_process_index,
                shuffle=False,
            ),
        )
        accelerator = Accelerator()
        for pred, label in tqdm(data_loader, position=args.local_process_index):
            wer_scores = wer.compute(predictions=pred, references=label)
            cer_scores = cer.compute(predictions=pred, references=label)
            wer_score_ls.append(wer_scores)
            cer_score_ls.append(cer_scores)
            result_ls.append((pred, label, wer_scores, cer_scores))

        if args.parallel_mode != "not_distributed":
            wer_score_ls = accelerator.gather_for_metrics(wer_score_ls)
            cer_score_ls = accelerator.gather_for_metrics(cer_score_ls)
            result_ls = accelerator.gather_for_metrics(result_ls)

        for pred, label, wer_scores, cer_scores in result_ls:
            predictions_table.add_data(pred, label, wer_scores, cer_scores)
    else:
        wer_scores = wer.compute(predictions=src, references=lbl)
        cer_scores = cer.compute(predictions=src, references=lbl)

    return {
        "wer": np.mean(wer_score_ls) if wer_score_ls else wer_scores,
        "cer": np.mean(cer_score_ls) if cer_score_ls else cer_scores,
        "predictions_result": predictions_table,
    }


METRICS_REGISTRY = {
    "tnt_dual": compute_tnt_metrics,
    "p2s_tnt": compute_tnt_metrics,
    "s2p_tnt": compute_tnt_metrics,
    "all_tnt": compute_tnt_metrics,
}
