from transformers import PreTrainedTokenizer, TrainingArguments


def default_preprocessor(example, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    preprocess_finish_ls = list()
    for conversations in example["conversations"]:
        text = tokenizer.apply_chat_template(conversations)
        outputs = tokenizer(text, return_tensors="np", return_length=True)

        finish_data = {
            "input_ids": outputs.input_ids,
            args.length_column_name: outputs.length,
        }
        preprocess_finish_ls.append(finish_data)

    return_dict = dict()
    for res in preprocess_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict
