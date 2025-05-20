import numpy as np
import wandb
from evaluate import load

from transformers import EvalPrediction, PreTrainedTokenizer, TrainingArguments


def compute_tnt_metrics(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizer,
    args: TrainingArguments,
) -> dict[str, float]:
    eval_pred.predictions[eval_pred.predictions == -100] = tokenizer.pad_token_id
    preds, labels = (
        [
            x.replace(tokenizer.pad_token, "").split("<|im_start|>")[-1].replace("<|im_end|>", "")
            for x in tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=False)
        ],
        [
            x.replace(tokenizer.pad_token, "")
            for x in tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=False)
        ],
    )

    wer_score_ls, cer_score_ls = [], []
    predictions_table = wandb.Table(columns=["Prediction", "Label", "WER", "CER"])
    wer, cer = load("evaluate-metric/wer"), load("evaluate-metric/cer")

    for pred, label in zip(preds, labels):
        wer_scores = wer.compute(predictions=preds, references=labels)
        cer_scores = cer.compute(predictions=preds, references=labels)
        wer_score_ls.append(wer_scores)
        cer_score_ls.append(cer_scores)
        predictions_table.add_data(pred, label, wer_scores, cer_scores)

    return {
        "wer": np.mean(wer_score_ls),
        "cer": np.mean(cer_score_ls),
        "predictions_result": predictions_table,
    }


METRICS_REGISTRY = {
    "tnt_dual": compute_tnt_metrics,
    "p2s_tnt": compute_tnt_metrics,
    "s2p_tnt": compute_tnt_metrics,
    "all_tnt": compute_tnt_metrics,
}
