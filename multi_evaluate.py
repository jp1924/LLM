import argparse
import json
import re
from pathlib import Path
from time import sleep
from typing import List, Tuple

import tiktoken
from datasets import Dataset, load_from_disk
from openai import AzureOpenAI, OpenAI


JUDGE_TEMPLATE = {
    "single_turn": """너는 질문에 대한 한국어 언어 모델의 답변을 매우 꼼꼼히 평가할 것이다. 공정한 평가를 위해 아래의 규칙을 준수한다.

# 기본 규칙
1. 질문의 요구사항을 충분히 반영하였는지 상세히 분석할 것.
2. 답변 과정에서 누락되었거나 포함되지 못하여 아쉬운 부분에 대하여 상세히 분석할 것.
3. 답변의 길이가 평가 결과에 영향을 미치지 않도록 할 것.
4. Additional Reference가 제공된다면 평가 시 해당 정보를 참고할 것.

# 언어 요구사항
- 모델은 반드시 한국어로 답변해야 하며, 다른 언어로의 답변은 절대 허용되지 않는다.
- 예외적으로 질문이 영어로 답변할 것을 요구할 때에만 영어 답변이 허용된다.
- 한국어로 답변하지 않을 경우, 점수는 0점 처리된다.
- 언어 요구사항을 충족하는 것은 필수적이나, 이 요구사항의 충족이 답변의 질적 평가에 추가 점수로 이어지지는 않는다.

# 평가 출력 방식
**주어진 Question에 집중하여** Model's Response에 대한 평가와 1~10의 점수를 부여한다. 답변에 대한 평가는 4~5 문장으로 규칙을 참고하여 상세히 작성한다.

# 출력 형식
평가: 평가 내용
점수: 숫자""",
    "multi_turn": """너는 대화 후 이어지는 후속 질문에 대한 한국어 언어 모델의 답변을 매우 꼼꼼히 평가할 것이다. 공정한 평가를 위해 아래의 규칙을 준수한다.

# 기본 규칙
1. 질문의 요구사항을 충분히 반영하였는지 상세히 분석할 것.
2. 답변 과정에서 누락되었거나 포함되지 못하여 아쉬운 부분에 대하여 상세히 분석할 것.
3. 답변의 길이가 평가 결과에 영향을 미치지 않도록 할 것.
4. Additional Reference가 제공된다면 평가 시 해당 정보를 참고할 것.
5. 후속 질문에 대한 답변이 이전 대화 맥락과 일치하는지 확인할 것.

# 언어 요구사항
- 모델은 반드시 한국어로 답변해야 하며, 다른 언어로의 답변은 절대 허용되지 않는다.
- 예외적으로 질문이 영어로 답변할 것을 요구할 때에만 영어 답변이 허용된다.
- 한국어로 답변하지 않을 경우, 점수는 0점 처리된다.
- 언어 요구사항을 충족하는 것은 필수적이나, 이 요구사항의 충족이 답변의 질적 평가에 추가 점수로 이어지지는 않는다.

# 평가 출력 방식
**주어진 Question에 집중하여** Model's Response에 대한 평가와 1~10의 점수를 부여한다. 답변에 대한 평가는 4~5 문장으로 규칙을 참고하여 상세히 작성한다.

# 출력 형식
평가: 평가 내용
점수: 숫자""",
}


use_azure = False


client = AzureOpenAI() if use_azure else OpenAI()


def num_tokens_from_messages(messages, model):
    "Return the number of tokens used by a list of messages."
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def evaluate_eval_result(example, judge_model, eval_repeat_num=5):
    def send_request_to_gpt(
        message: List[dict],
        model: str,
        seed: int = 42,
        send_retry: int = 10,
        error_interval_time: int = 10,
    ) -> Tuple[str, str, float]:
        judge_score = 0.0
        for retries in range(send_retry):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=0.0,
                    n=1,
                    response_format={"type": "text"},
                    seed=seed,
                )
                judge_result = response.choices[0].message.content

                judge_reason_match = re.search(r"평가:(.*?)점수:", judge_result.replace("*", ""), re.DOTALL)
                judge_reason = judge_reason_match.group(1).strip() if judge_reason_match else "No judge message found"
                judge_score_match = re.search(r"점수:\s*(\d+(\.\d+)?)", judge_result.replace("*", ""))
                if judge_score_match:
                    judge_score = float(judge_score_match.group(1))
                else:
                    raise ValueError("No score found in response")

                break
            except BaseException as e:
                print(f"{retries}회의 instruction 생성 중 {e}과 같은 애러가 발생해 다시 retry함.")
                sleep(error_interval_time)
        else:
            print("Impossible prompt, aborting..!")
            judge_result = ""
            judge_reason = "Impossible to judge due to repetition."
            judge_score = 0.0

        return judge_result, judge_reason, judge_score

    finish_single_judge_reason_ls, finish_single_judge_score_ls = list(), list()
    finish_double_judge_reason_ls, finish_double_judge_score_ls = list(), list()
    finish_prompt_token_num_ls, finish_answer_token_num_ls = list(), list()
    for questions, outputs, references in zip(example["questions"], example["outputs"], example["references"]):
        # questions, outputs, references으로 prompt 생성
        start_prompt, end_prompt = (
            "아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. 특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.",
            "\n\n[[대화 종료. 평가 시작.]]",
        )
        single_question, double_question = (
            f"\n\n**Question**\n{questions[0]}",
            f"\n\n**Follow-up Question.**\n{questions[1]}",
        )
        single_output, double_output = (
            f"\n\n**Model's Response**\n{outputs[0]}",
            f"\n\n**Model's Response**\n{outputs[1]}",
        )
        single_reference, double_reference = (
            f"\n\n**Additional Reference**\n{references[0]}" if references[0] else "",
            f"\n\n**Additional Reference**\n{references[1]}" if references[1] else "",
        )

        single_turn_eval = start_prompt + single_question + single_reference + single_output + end_prompt
        double_turn_eval = (
            start_prompt
            + single_question
            + single_reference
            + single_output
            + double_question
            + double_reference
            + double_output
            + end_prompt
        )

        # eval_repeat_num만큼 반복하여 평가

        prompt_token_num, answer_token_num = 0, 0
        single_judge_reason_ls, single_judge_score_ls = list(), list()
        double_judge_reason_ls, double_judge_score_ls = list(), list()
        for _ in range(eval_repeat_num):
            single_prompt_message = [
                {"role": "system", "content": JUDGE_TEMPLATE["single_turn"]},
                {"role": "user", "content": single_turn_eval},
            ]
            double_prompt_message = [
                {"role": "system", "content": JUDGE_TEMPLATE["multi_turn"]},
                {"role": "user", "content": double_turn_eval},
            ]
            single_full_text, single_judge_reason, single_judge_score = send_request_to_gpt(
                single_prompt_message, model=judge_model
            )
            double_full_text, double_judge_reason, double_judge_score = send_request_to_gpt(
                double_prompt_message, model=judge_model
            )
            single_answer_message = [{"role": "assistant", "content": single_full_text}]
            double_answer_message = [{"role": "assistant", "content": double_full_text}]

            prompt_token_num += num_tokens_from_messages(single_prompt_message, judge_model)
            prompt_token_num += num_tokens_from_messages(double_prompt_message, judge_model)

            answer_token_num += num_tokens_from_messages(single_answer_message, judge_model)
            answer_token_num += num_tokens_from_messages(double_answer_message, judge_model)

            single_judge_reason_ls.append(single_judge_reason)
            single_judge_score_ls.append(single_judge_score)

            double_judge_reason_ls.append(double_judge_reason)
            double_judge_score_ls.append(double_judge_score)

        finish_single_judge_reason_ls.append(single_judge_reason_ls)
        finish_single_judge_score_ls.append(single_judge_score_ls)

        finish_double_judge_reason_ls.append(double_judge_reason_ls)
        finish_double_judge_score_ls.append(double_judge_score_ls)

        finish_prompt_token_num_ls.append(prompt_token_num)
        finish_answer_token_num_ls.append(answer_token_num)

    example["single_judge_reason"] = finish_single_judge_reason_ls
    example["single_judge_score"] = finish_single_judge_score_ls
    example["double_judge_reason"] = finish_double_judge_reason_ls
    example["double_judge_score"] = finish_double_judge_score_ls
    example["prompt_token_num"] = finish_prompt_token_num_ls
    example["answer_token_num"] = finish_answer_token_num_ls

    return example


def main(eval_jsonl_dir: Path, judge_model: str = "gpt-4o-mini-2024-07-18", eval_repeat_num: int = 5) -> None:
    category_ls = [
        "이해(Understanding)",
        "글쓰기(Writing)",
        "추론(Reasoning)",
        "문법(Grammar)",
        "코딩(Coding)",
        "수학(Math)",
    ]
    eval_jsonl_dir = eval_jsonl_dir.joinpath("generated")
    eval_json_ls = sorted(eval_jsonl_dir.glob("*.jsonl"))

    for eval_jsonl_path in eval_json_ls:
        eval_save_dir = eval_jsonl_path.parent.joinpath(f"evaluate_result-{eval_jsonl_path.stem}")

        if not eval_save_dir.exists():
            with open(eval_jsonl_path.as_posix(), mode="r", encoding="utf-8-sig") as f:
                eval_dataset = [json.loads(line.strip()) for line in f.readlines()]
                eval_dataset = Dataset.from_list(eval_dataset)

            eval_dataset = eval_dataset.map(
                evaluate_eval_result,
                batch_size=2,
                batched=True,
                keep_in_memory=True,
                num_proc=10,
                fn_kwargs={
                    "judge_model": judge_model,
                    "eval_repeat_num": eval_repeat_num,
                },
            )
            eval_dataset.save_to_disk(eval_save_dir.as_posix())

        eval_dataset = load_from_disk(eval_save_dir.as_posix())

        total_prompt_token_num, total_answer_token_num = (
            sum(eval_dataset["prompt_token_num"]),
            sum(eval_dataset["answer_token_num"]),
        )

        table_row_ls, json_ls = list(), list()
        table_row_ls.append("| Category | Single turn | Multi turn | Single std | Double std |")
        table_row_ls.append("|---|---|---|---|---|")

        single_total_score_ls, double_total_score_ls, single_total_std_ls, doubel_total_std_ls = (
            list(),
            list(),
            list(),
            list(),
        )
        for category in category_ls:
            category_eval_dataset = eval_dataset.filter(lambda x: x["category"] == category, keep_in_memory=True)
            single_judge_score_ls = [
                score for score_ls in category_eval_dataset["single_judge_score"] for score in score_ls
            ]
            double_judge_score_ls = [
                score for score_ls in category_eval_dataset["double_judge_score"] for score in score_ls
            ]

            single_mean_score = sum(single_judge_score_ls) / len(single_judge_score_ls)
            single_std_score = sum([(score - single_mean_score) ** 2 for score in single_judge_score_ls]) / len(
                single_judge_score_ls
            )

            double_mean_score = sum(double_judge_score_ls) / len(double_judge_score_ls)
            double_std_score = sum([(score - double_mean_score) ** 2 for score in double_judge_score_ls]) / len(
                single_judge_score_ls
            )

            json_ls.append(
                {
                    "category": category,
                    "single_mean_score": single_mean_score,
                    "double_mean_score": double_mean_score,
                    "single_std_score": single_std_score,
                    "double_std_score": double_std_score,
                }
            )

            single_total_score_ls.append(single_mean_score)
            double_total_score_ls.append(double_mean_score)
            single_total_std_ls.append(single_std_score)
            doubel_total_std_ls.append(double_std_score)

            table_row_ls.append(
                f"| {category} | {round(single_mean_score, 2):.2f} | {round(double_mean_score, 2):.2f} | {round(single_std_score, 2)} | {round(double_std_score, 2)} |"
            )
        table_row_ls.append(
            f"| total | {round(sum(single_total_score_ls) / len(single_total_score_ls), 2):.2f} | {round(sum(double_total_score_ls) / len(double_total_score_ls), 2):.2f} | {round(sum(single_total_std_ls) / len(single_total_std_ls), 2)} | {round(sum(doubel_total_std_ls) / len(doubel_total_std_ls), 2)} |"
        )
        eval_save_dir.joinpath("result.md").write_text("\n".join(table_row_ls))
        eval_save_dir.joinpath("result.json").write_text(json.dumps(json_ls, indent=4, ensure_ascii=False))
        eval_save_dir.joinpath("token_num.json").write_text(
            json.dumps(
                {"prompt_token_num": total_prompt_token_num, "answer_token_num": total_answer_token_num},
                indent=4,
                ensure_ascii=False,
            )
        )


if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_checkpoint_ls", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--eval_repeat_num", type=int, default=5)
    args = parser.parse_args()

    eval_checkpoint_ls = sorted(Path(args.eval_checkpoint_ls).glob("*"))
    for checkpoint_dir in eval_checkpoint_ls:
        main(checkpoint_dir, args.judge_model, args.eval_repeat_num)
