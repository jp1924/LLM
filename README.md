# LLM

LLM 학습시키는 코드 모와둔 repo

## packing

![packing attention_mask](assets/packing_attention_mask.png) <br/>
대략 packing해서 들어가면 attention_mask가 이런식으로 들어가게 됨. <br/>
근데 flash_attention은 attention_mask 따로 안주고, position_ids로 분간함. <br/>

## refer

- [DataCollatorWithFlatteningtransformers.DataCollatorWithFlattening](https://huggingface.co/docs/transformers/ko/main_classes/data_collator#transformers.DataCollatorWithFlattening)
- [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2)
