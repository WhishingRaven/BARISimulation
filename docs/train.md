# 학습

학습 코드는 `bari_sim/train/`에 있습니다. Pipeline은 simulation rollout에서
trajectory metric을 수집하고, task objective가 이를 scalar fitness로 변환한 뒤,
선택한 algorithm에 전달합니다. Algorithm은 task별 성공이나 충돌 판정을 알지
않습니다. 새 algorithm은 `bari_sim/train/algorithms/`에 추가하고 pipeline의
dispatch에 등록합니다.

## CEM

현재 지원하는 algorithm은 cross-entropy method(CEM)뿐이며
`--algorithm cem`이 기본값입니다. CEM은 policy parameter 분포에서 population을
sampling하고, episode fitness가 높은 elite로 평균과 표준편차를 갱신합니다. 전체
세대에서 가장 좋은 policy를 기존 `infer`와 `evaluate`가 읽는 JSON 형식으로
저장합니다.

학습은 기본적으로 headless입니다. `--render`는 rollout을 보여주기만 하며
metric이나 fitness 계산을 바꾸지 않습니다. 같은 seed와 설정은 같은 parameter
sampling 순서를 사용합니다.

## 명령 예제

아래 block은 zsh에 그대로 복사해 실행할 수 있습니다.

### 실제 CEM 학습

```zsh
setopt interactivecomments
train_args=(
  --robots '2*5'                  # Required: Robot grid (2 rows × 5 columns)
  --task collision-avoidance      # Required: Task to train
  --difficulty 1                  # Required: Task difficulty (1-5)
  --algorithm cem                 # Optional: Algorithm; default is cem
  --generations 30                # Optional: CEM generations; default is 5
  --population 32                 # Optional: Candidates per generation; default is 8
  --elite-fraction 0.25           # Optional: Elite fraction; default is 0.25
  --initial-std 0.75              # Optional: Initial parameter std; default is 0.75
  --min-std 0.05                  # Optional: Minimum parameter std; default is 0.05
  --episodes-per-candidate 1      # Optional: Episodes per candidate; default is 1
  --duration 120                  # Optional: Episode limit in seconds; default is 60
  --seed 7                        # Optional: Random seed; default is 7
  --output models/cem/ca-d1.json  # Optional: Output; default is timestamped JSON
)
barisimulation train "${train_args[@]}"
unset train_args
```

학습 중 `--render`는 선택 옵션이지만 성능이 크게 느려지므로 보통 생략합니다.

### 빠른 smoke test

Pipeline과 모델 저장이 작동하는지만 짧게 확인합니다. 이 설정으로 유용한 policy가
학습된다고 기대할 수 없습니다.

```zsh
setopt interactivecomments
smoke_args=(
  --robots '2*5'                      # Required: Robot grid (2 rows × 5 columns)
  --task collision-avoidance          # Required: Task to train
  --difficulty 1                      # Required: Task difficulty (1-5)
  --generations 2                     # Optional: Small generation count for a quick check
  --population 4                      # Optional: Small population for a quick check
  --duration 5                        # Optional: Short episode duration for a quick check
  --output models/cem/smoke-test.json # Optional: Explicit smoke-test model path
)
barisimulation train "${smoke_args[@]}"
unset smoke_args
```

### 저장한 모델 실행

```zsh
setopt interactivecomments
infer_args=(
  --robots '2*5'                 # Required: Robot grid used for inference
  --task collision-avoidance     # Required: Task matching the saved model
  --model models/cem/ca-d1.json  # Required: Saved model path
  --render                       # Optional: Open viewer; headless when omitted
)
barisimulation infer "${infer_args[@]}"
unset infer_args
```

### 여러 episode 평가

```zsh
setopt interactivecomments
evaluate_args=(
  --robots '2*5'                 # Required: Robot grid used for evaluation
  --task collision-avoidance     # Required: Task matching the saved model
  --difficulty 1                 # Required: Evaluation difficulty (1-5)
  --model models/cem/ca-d1.json  # Required: Saved model path
  --episodes 5                   # Optional: Episode count; default is 1
  --duration 120                 # Optional: Episode limit; default is 120 seconds
)
barisimulation evaluate "${evaluate_args[@]}"
unset evaluate_args
```

## collision-avoidance objective

이 task의 fitness는 정규화된 항을 가중 합산합니다.

```text
fitness = progress_reward + success_bonus
          - cohesion_penalty - time_penalty
          - collision_penalty - flipped_penalty
```

기본 가중치는 progress 3, success 5, cohesion 1, time 0.5, collision 2,
flipped/immobile 3입니다. 분산은 0.10 m², 충돌은 로봇당 event 1회를 기준으로
정규화합니다. 이 값들은 `CollisionAvoidanceObjectiveConfig`에서 변경할 수 있습니다.

Progress는 `(초기 목표 거리 - 최종 목표 거리) / 초기 목표 거리`입니다. 이동하지
않으면 0, 목표에서 멀어지면 음수, 도달하면 약 1입니다. Cohesion은 최종 frame이
아닌 episode 전체의 평균 위치 분산을 사용합니다. 충돌은 지속 접촉 frame 수가
아니라 `not colliding`에서 `colliding`으로 바뀐 event 수입니다. 실패는 조기 종료해도
시간 penalty 전부를 받으므로 충돌이나 뒤집힘으로 빨리 끝나는 policy가 유리하지
않습니다.

각 세대 로그에는 best/mean/elite fitness, parameter 표준편차, best candidate의
objective component가 포함됩니다. `infer`, `evaluate`, `train`은 같은 task objective를
사용합니다.
