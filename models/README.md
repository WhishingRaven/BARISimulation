# Models

`barisimulation train`이 생성한 JSON policy를 저장하는 디렉터리입니다.

기본 파일명은 다음과 같습니다.

- `collision-avoidance.json`
- `gap.json`
- `step.json`

저장 파일에는 observation feature 순서, 세 action head의 weight, task/difficulty/robot-grid metadata가 포함됩니다. 저장된 학습 결과는 repository에 기본 포함하지 않습니다.
