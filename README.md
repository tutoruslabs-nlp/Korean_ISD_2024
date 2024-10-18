# 문장 연결 추론 Baseline
본 리포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '부적절 발언 탐지'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.  

학습 및 추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.  

|Model|Accuracy(%)|
|:---|---:|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (without SFT)|61.5|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (with SFT)|90.7|

## 리포지토리 구조 (Repository Structure)
```
# 학습에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
├── test.py
└── train.py

# 학습에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
[
  {
    "id": "nikluge-2024-au-dev-00006",
    "input": {
      "document_id": "nikluge-2024-au-dev-00006",
      "sentences": [
        {
          "id": "nikluge-2024-au-dev-00006-001",
          "sentence": "우쩌면도 롱샷이랑."
        },
        {
          "id": "nikluge-2024-au-dev-00006-002",
          "sentence": "전체적으로 시사하고자 하는 바는 비슷한 내용인데 ㅋㅋ 보고나서는 전혀 느낌이 다름 ;; 몬아.."
        },
        {
          "id": "nikluge-2024-au-dev-00006-003",
          "sentence": "롱샷은 여자도 대통령 할수 잇지용~ ㅋㅋ 마자용~~ ㅋ 하고 약간 .."
        },
        {
          "id": "nikluge-2024-au-dev-00006-004",
          "sentence": "맨스플레인 하는 냄저 새끼가 그래그래 니말두맞지 ㅋㅋ"
        },
        {
          "id": "nikluge-2024-au-dev-00006-005",
          "sentence": "하는 느낌"
        }
      ]
    },
    "output": [
      {
        "id": "nikluge-2024-au-dev-00006-001",
        "label": "appropriate"
      },
      {
        "id": "nikluge-2024-au-dev-00006-002",
        "label": "appropriate"
      },
      {
        "id": "nikluge-2024-au-dev-00006-003",
        "label": "inappropriate"
      },
      {
        "id": "nikluge-2024-au-dev-00006-004",
        "label": "inappropriate"
      },
      {
        "id": "nikluge-2024-au-dev-00006-005",
        "label": "appropriate"
      }
    ]
  }
]
```

## 실행 방법 (How to Run)
### 학습 (Train)
```
CUDA_VISIBLE_DEVICES=0 python -m run.train \
    --trainset resource/data/nikluge-2024-au-train.json \
    --devset resource/data/nikluge-2024-au-dev.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --epoch 1 \
    --lr 2e-5 \
    --batch_size 4 \
    --warmup_steps 20 \
    --gradient_accumulation_steps 16 \
    --save_dir ./models/e1
```

### 추론 (Inference)
```
python -m run.test \
    --input resource/data/nikluge-2024-au-test.json \
    --output result.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
```


## Reference
huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
