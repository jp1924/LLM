from evaluate import load

from transformers import EvalPrediction, PreTrainedTokenizer


def compute_tnt_metrics(eval_pred: EvalPrediction, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
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

    bleu, rouge = load("evaluate-metric/bleu"), load("evaluate-metric/rouge")
    bleu_scores = bleu.compute(predictions=preds, references=labels, tokenizer=lambda x: tokenizer.tokenize(x))
    rouge_scores = rouge.compute(predictions=preds, references=labels, tokenizer=lambda x: tokenizer.tokenize(x))

    return {"bleu": bleu_scores["bleu"], **rouge_scores}


METRICS_REGISTRY = {
    "tnt_dual": compute_tnt_metrics,
    "p2s_tnt": compute_tnt_metrics,
    "s2p_tnt": compute_tnt_metrics,
    "all_tnt": compute_tnt_metrics,
}
