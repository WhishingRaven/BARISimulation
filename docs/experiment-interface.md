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

명시적으로 이 명령을 실행할 때만 policy search가 시작됩니다. 현재 구현된 알고리즘은 cross-entropy method(CEM)이며, `--algorithm cem`이 기본값입니다. 기본 모델 경로는 `models/cem/<UTC 시간>.json`입니다. 구현된 trainer는 선언된 local observation만 사용하는 공유 linear policy를 CEM으로 탐색합니다. 진행 로그는 stderr, 최종 결과 JSON은 stdout으로 출력됩니다.

긴 실험의 범위는 옵션으로 결정합니다.

```bash
barisimulation train \
  --robots '4*5' \
  --task gap \
  --difficulty 4 \
  --generations 20 \
  --population 32 \
  --elite-fraction 0.25 \
  --initial-std 0.75 \
  --min-std 0.05 \
  --episodes-per-candidate 2 \
  --duration 120 \
  --seed 7 \
  --output models/cem/gap-d4.json
```

세부 학습 구조와 collision-avoidance objective는 [train.md](train.md)를 참고합니다.

## Infer

```bash
barisimulation infer \
  --robots '4*5' \
  --model models/cem/gap-d4.json \
  --task gap \
  --render
```

`--difficulty`을 생략하면 model metadata의 난이도를 사용합니다. `--render`가 없으면 headless로 실행합니다. `--viewer`는 기존 호환을 위한 `--render`의 별칭입니다.

## Evaluate

```bash
barisimulation evaluate --robots '4*5' --task gap --difficulty 4 --model models/cem/gap-d4.json
```

시간 기반으로 저장된 모델 중 무엇을 평가할지 자동으로 정할 수 없으므로 `--model`은 필수입니다.

```bash
barisimulation evaluate \
  --robots '4*5' \
  --task gap \
  --difficulty 4 \
  --model models/cem/gap-d4.json \
  --episodes 5 \
  --duration 120 \
  --render
```

출력 JSON에는 task 성공 여부, 성공 로봇 수/비율, 위치 분산, 충돌, 뒤집힘, task별 치수와 score가 포함됩니다.
