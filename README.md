# BARISimulation

MuJoCo 기반 BARI 군집 로봇 시뮬레이터입니다. 프로젝트 전체는 다음 세 task만 다룹니다.

- `collision-avoidance`: 군집을 유지하며 목표 지점까지 이동
- `gap`: 협동하여 대부분의 로봇이 틈을 통과
- `step`: 협동하여 대부분의 로봇이 단차 위로 이동

로봇의 고정 형상은 10 cm × 15 cm × 0.5 cm, 50 g이며 policy step은 0.5초입니다. 물리 적분은 더 작은 timestep으로 수행하지만 observation과 action 교환은 정확히 0.5초마다 일어납니다.

## 설치

이름이 `barisimulation`인 conda 환경을 사용합니다.

```bash
conda env create -f environment.yml
conda activate barisimulation
python -m pip install -e .
```

이미 같은 이름의 환경이 있다면 다음처럼 갱신합니다.

```bash
conda env update -n barisimulation -f environment.yml --prune
conda activate barisimulation
python -m pip install -e .
```

## 명령어

```bash
barisimulation help
barisimulation help manual
barisimulation manual --robots '1*1' --environment flat
barisimulation infer --robots '2*5' --model models/cem/<timestamp>.json --task collision-avoidance
barisimulation train --robots '2*5' --task gap --difficulty 1
barisimulation evaluate --robots '2*5' --task gap --difficulty 1 --model models/cem/<timestamp>.json
```

현재 학습 알고리즘은 CEM이며 `--algorithm cem`이 기본값입니다. `train`의 기본 출력은 `models/cem/<UTC 시간>.json`입니다. 시간 기반 모델은 어느 파일을 평가할지 추측할 수 없으므로 `evaluate`에서는 `--model`이 필수입니다. 세 명령 모두 기본적으로 headless로 실행하며, `--render`를 추가하면 MuJoCo 창에서 rollout을 볼 수 있습니다. 진행 로그는 stderr, 최종 결과 JSON은 stdout으로 출력합니다.

## 문서

- [로봇 상세 사양](docs/robot-specification.md)
- [Action과 observation](docs/action-observation.md)
- [Task와 난이도](docs/tasks.md)
- [수동 조작](docs/manual-control.md)
- [실험 명령 인터페이스](docs/experiment-interface.md)

## 디렉터리

```text
bari_sim/
  robot/       고정 사양, discrete action, local observation
  simulation/  MJCF 장면, 물리 step, 센서, 가시 attachment
  tasks/       난이도 표와 task 성공/평가 지표
  policies/    저장 가능한 decentralized policy
  workflows/   manual, train, infer, evaluate 실행 흐름
  cli.py       barisimulation 명령 라우팅
docs/          로봇·action·task·실험 계약 문서
models/        train이 생성하는 policy JSON
tests/         고정 사양과 물리 action 검증
```
