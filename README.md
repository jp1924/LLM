# LLM

LLM 학습시키는 코드 모와둔 repo

## packing

![packing attention_mask](assets/packing_attention_mask.png) <br/>
대략 packing해서 들어가면 attention_mask가 이런식으로 들어가게 됨. <br/>
근데 flash_attention은 attention_mask 따로 안주고, position_ids로 분간함. <br/>

## LogicKor

> 아직 제작 중
대충 학습 끝난 뒤 LogicKor 돌리기 귀찮아서 만든 코드<br/>
학습 중 chekcpoint save하고 난 뒤 수행함.<br/>
근데 zero-3에선 config 설정에 따라 eval 하는데 4시간 걸리더라<br/>

## refer

- [DataCollatorWithFlatteningtransformers.DataCollatorWithFlattening](https://huggingface.co/docs/transformers/ko/main_classes/data_collator#transformers.DataCollatorWithFlattening)
- [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2)
