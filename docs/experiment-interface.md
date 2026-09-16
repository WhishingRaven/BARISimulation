# 실험 명령 인터페이스

## 공통 확인

```bash
barisimulation help
barisimulation help train
barisimulation help infer
barisimulation help evaluate
```

## Train

```bash
barisimulation train --robots '2*5' --task collision-avoidance --difficulty 1
```

명시적으로 이 명령을 실행할 때만 policy search가 시작됩니다. 기본 모델 경로는 `models/<task>.json`입니다. 구현된 trainer는 선언된 local observation만 사용하는 공유 linear policy를 cross-entropy method로 탐색합니다.

긴 실험의 범위는 옵션으로 결정합니다.

```bash
barisimulation train \
  --robots '4*5' \
  --task gap \
  --difficulty 4 \
  --generations 20 \
  --population 32 \
  --duration 120 \
  --seed 7 \
  --output models/gap-d4.json
```

## Infer

```bash
barisimulation infer \
  --robots '4*5' \
  --model models/gap-d4.json \
  --task gap \
  --viewer
```

`--difficulty`을 생략하면 model metadata의 난이도를 사용합니다. `--viewer`가 없으면 headless JSON 결과를 출력합니다.

## Evaluate

```bash
barisimulation evaluate --robots '4*5' --task gap --difficulty 4
```

`--model`이 없으면 `models/gap.json`을 읽습니다. 다른 파일은 다음처럼 지정합니다.

```bash
barisimulation evaluate \
  --robots '4*5' \
  --task gap \
  --difficulty 4 \
  --model models/gap-d4.json \
  --episodes 5 \
  --duration 120
```

출력 JSON에는 task 성공 여부, 성공 로봇 수/비율, 위치 분산, 충돌, 뒤집힘, task별 치수와 score가 포함됩니다.
