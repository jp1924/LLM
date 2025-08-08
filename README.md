# LLM

LLM과 LVM에 SFT, GRPO, Pretrain하고 전처리하는 코드를 정리한 github repo<br>

## running

```
git clone https://github.com/jp1924/LLM.git
docker compose up -d 
```

Docker와 Compose Version: 27.3.1-build ce12230, v2.29.7<br>
근데 package 버전이 빠르게 바뀌어서 실행 안되면 Dockerfile 수정할 필요가 있음<br>

## Utils

flash-attn을 pypi에서 다운받아서 설치하면 컴파일 하는데 많은 시간이 걸리지만<br>
flash-attn repo에서 제공하는 whl파일은 이미 컴파일이 되어 있기 때문에 설치하는데 많은 시간이 걸리지 않는다.<br>

whl로 설치하는 경우 다음 명령어를 입력해서 개발환경의 버전을 확인할 필요가 있따.<br>
> python -c "import torch; print(f'torch version: {torch.__version__}'); print(f'cuda version: {torch.version.cuda}'); print(f'abi version: {torch.compiled_with_cxx11_abi()}')

이후 flash-attn repo가서 본인 환경에 맞는 flash-attn 설치하면 된다.<br>
