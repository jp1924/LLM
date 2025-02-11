from transformers import PreTrainedTokenizer, TrainingArguments


def check_special_token(input_ids, tokenizer):
    if input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids = input_ids + [tokenizer.eos_token_id]

    return input_ids


def sft_processor(example, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    preprocess_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416
        text = tokenizer.apply_chat_template(row_dataset["conversation"], tokenize=False)
        outputs = tokenizer(text, return_tensors="np", return_length=True)

        preprocess_finish_ls.append(
            {
                "input_ids": check_special_token(outputs.input_ids[0], tokenizer),
                args.length_column_name: outputs.length[0],
            }
        )

    return_dict = dict()
    for res in preprocess_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def pretrain_processor(example, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416

        if "sentence_ls" in row_dataset:
            for sentence in row_dataset["sentence_ls"]:
                outputs = tokenizer(sentence, return_length=True)
                finish_data = {
                    "input_ids": check_special_token(outputs.input_ids, tokenizer),
                    args.length_column_name: outputs.length[0],
                }
                process_finish_ls.append(finish_data)
        else:
            outputs = tokenizer(row_dataset["sentence"], return_length=True)
            finish_data = {
                "input_ids": check_special_token(outputs.input_ids, tokenizer),
                args.length_column_name: outputs.length[0],
            }
            process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def reasoning_processor(example, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


PROCESSOR_REGISTRY = {
    "sft": sft_processor,
    "reasoning_sft": reasoning_processor,
    "pretrain": pretrain_processor,
}
