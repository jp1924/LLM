# SFT

SFT, Pretrain, TNT모델 학습시키는 코드가 모여저 있는 폴더

## file description

```ascii
sft
 ┣ callbacks.py # callback 함수 모와둔 파일, 근데 별로 사용하진 않는다.
 ┣ main.py # 데이터 전처리, 모델, callback, Trainer 불러와서 학습을 시작하는 파일
 ┣ metrics.py # TNT 한정으로 evaluate 평가할 때 활용하는 파일
 ┣ optimization.py # 내가 정의한 scheduler 정의한 파일
 ┣ preprocessor.py # 데이터 전처리 코드 모와둔 파일
 ┗ trainer.py # SPFHP Packing Sampler를 적용한 Trainer와 Collator가 정의된 파일
              # evaluate시 model.generate 수행할 수 있게 Trainer를 수정함.
```

## Data Preprocess

pretrain는 `sentence`, `sentence_ls`<br>
sft는 `conversations`컬럼만 받도록 설계되어 있다.<br>

컬럼명이 맞지 않는 데이터의 경우 Datasets의 BuilderScript를 활용해서 컬럼명을 맞춘 뒤 작업한다.<br>
그리고 builder script는 [HF_builde](https://github.com/jp1924/HF_builders)에 올려뒀다.<br>
