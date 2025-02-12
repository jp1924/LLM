import json
import shutil
from pathlib import Path
from typing import Tuple

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
)


CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if not continue_final_message is defined %}{% set continue_final_message = false %}{% endif %}{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>' }}{% if message.role == 'user' %}{{ '<User>' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ image_token }}{% elif content.type == 'text' %}{{ content.text }}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'system' %}{{ '<System>' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ image_token }}{% elif content.type == 'text' %}{{ content.text }}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'assistant' %}{{ '<Assistant>' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ image_token }}{% elif content.type == 'reason' %}{{ '<Think>' }}{{ content.text }}{{ '</Think>' }}{% elif content.type == 'text' %}{{ content.text }}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{% endif %}{% if not (continue_final_message and loop.last) %}{{ '<end_of_turn>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>' }}{{ '### Assistant:\n' }}{% elif not continue_final_message %}{{ eos_token }}{% endif %}"


def get_language(language_model_name_or_path) -> Tuple[AutoModel, AutoConfig, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(language_model_name_or_path)
    config = AutoConfig.from_pretrained(language_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path, padding_side="left")
    if tokenizer.pad_token == tokenizer.eos_token:
        print(f"tokenizer의 pad_token이 {tokenizer.pad_token}과 같이 되어 있어서 {tokenizer.unk_token}으로 변경함.")
        tokenizer.pad_token = tokenizer.unk_token

    return (model, config, tokenizer)


def insert_img_token_to_gemma_tokenizer(tokenizer: PreTrainedTokenizer) -> Tuple[PreTrainedTokenizer, int]:
    new_special_token_ls = [
        {
            "content": "<System>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        {
            "content": "<User>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        {
            "content": "<Assistant>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        {
            "content": "<Think>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        {
            "content": "</Think>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
    ]

    unused_prefix = "unused"
    unsed_token_ls = [vocab_row for vocab_row in tokenizer.get_added_vocab().items() if unused_prefix in vocab_row[0]]
    unused_ls = sorted(unsed_token_ls, key=lambda vocab_row: vocab_row[1])

    if not unused_ls:
        raise ValueError("unused token이 존재하지 않음.")

    unused_token, unused_idx = unused_ls[0]

    model_dir = Path(tokenizer.vocab_file).parent
    tokenizer_config = json.loads(model_dir.joinpath("tokenizer_config.json").read_text())
    special_tokens_map = json.loads(model_dir.joinpath("special_tokens_map.json").read_text())
    tokenizer_raw_file = json.loads(model_dir.joinpath("tokenizer.json").read_text())

    for (unused_token, unused_idx), new_special_token in zip(unused_ls, new_special_token_ls):
        tokenizer_config["added_tokens_decoder"][str(unused_idx)] = new_special_token
        tokenizer_raw_file["added_tokens"][unused_idx] = {"id": unused_idx, **new_special_token}
        tokenizer_raw_file["model"]["vocab"].pop(unused_token)
        tokenizer_raw_file["model"]["vocab"][new_special_token["content"]] = unused_idx

    save_path = model_dir.joinpath("new_tokenizer")
    save_path.mkdir(exist_ok=True)

    save_path.joinpath("tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2, ensure_ascii=False))
    save_path.joinpath("special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2, ensure_ascii=False)
    )
    save_path.joinpath("tokenizer.json").write_text(json.dumps(tokenizer_raw_file, indent=2, ensure_ascii=False))

    new_tokenizer = AutoTokenizer.from_pretrained(save_path.as_posix())

    # NOTE: Remove the saved files
    shutil.rmtree(save_path)

    return new_tokenizer


def upload_to_hub(model: PreTrainedModel, processor: ProcessorMixin, hub_name: str, upload_retry: int = 10):
    for retries in range(upload_retry):
        try:
            model.push_to_hub(hub_name, private=True)
            processor.push_to_hub(hub_name, private=True)
        except BaseException as e:
            print(f"해당 애러가 {retries}시에 발생: {e}")
    else:
        exit("모델이 정상적으로 업로드 되질 않았음. 프로그램을 종료함.")


def main(
    language_model_name_or_path: str,
    output_dir: str,
    chat_template: str = CHAT_TEMPLATE,
    chat_template_forced: bool = False,
    push_to_hub: bool = False,
    upload_retry: int = 10,
):
    language_model, language_config, language_tokenizer = get_language(language_model_name_or_path)

    if "gemma" == language_config.model_type:
        language_tokenizer = insert_img_token_to_gemma_tokenizer(language_tokenizer)
    elif "gemma2" == language_config.model_type:
        language_tokenizer = insert_img_token_to_gemma_tokenizer(language_tokenizer)
    else:
        raise ValueError("지원하는 모델이 아님.")

    if chat_template_forced or language_tokenizer.chat_template is None:
        language_tokenizer.chat_template = chat_template
        print(f"tokenizer의 chat_template을 {chat_template}으로 변경")
    else:
        print(
            f"chat_template이 {language_tokenizer.chat_template}이라 따로 변경하지 않음. 변경하고 싶으면면 chat_template_forced를 True로 변경"
        )

    language_model.save_pretrained(output_dir)
    language_tokenizer.save_pretrained(output_dir)


if "__main__" in __name__:
    language_model_name_or_path = "google/gemma-2-9b"

    output_dir = "/root/output_dir/gemma-2-9b/R1"
    main(language_model_name_or_path, output_dir)
