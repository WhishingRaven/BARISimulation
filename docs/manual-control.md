# 수동 조작

```bash
conda activate barisimulation
barisimulation manual --robots '1*1' --environment flat
```

gap/step의 각 난이도를 확인할 수도 있습니다.

```bash
barisimulation manual --robots '2*5' --environment gap --difficulty 3
barisimulation manual --robots '2*5' --environment step --difficulty 2
```

## 키

| 키 | 동작 |
|---|---|
| W / S | curl / flatten |
| A / D | 본체 전체 좌회전 / 우회전 |
| C | motion 정지 |
| R / F | 앞쪽 flap 들기 / 내리기 |
| Space / X | 가시 attach / detach |
| 1–9, 0 | 로봇 1–10 직접 선택 |
| B / N | 이전 / 다음 로봇 선택 |
| P | 일시 정지 |
| Z | 초기 상태로 reset |

W/S/A/D는 누르고 있는 동안 0.5초 policy step을 연속 실행하고, 키를 놓으면 현재 step이 끝난 직후 정지합니다. 짧게 누르면 한 step만 실행됩니다. lift와 grip은 서로 독립적으로 latch되므로, 예를 들어 R을 누른 뒤 W를 누르면 앞쪽을 든 상태로 몸체를 curl합니다.

뷰어 왼쪽 위에는 이 조작법이 표시되고, 오른쪽 위에는 선택한 로봇의 `motion`, `lift`, `grip` 상태가 표시됩니다. 각 상태 앞의 채워진 점(`●`)이 현재 적용 중인 action component입니다.

macOS에서는 viewer 명령이 활성 conda 환경의 `mjpython`으로 자동 재실행됩니다. 따라서 사용자 명령은 그대로 `barisimulation manual ...`을 사용합니다.
