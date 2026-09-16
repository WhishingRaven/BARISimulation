# 수동 조작

```bash
conda activate barisimulation
barisimulation manual --robots 1*1 --environment flat
```

gap/step의 각 난이도를 확인할 수도 있습니다.

```bash
barisimulation manual --robots 2*5 --environment gap --difficulty 3
barisimulation manual --robots 2*5 --environment step --difficulty 2
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

선택된 로봇의 motion, lift, grip은 서로 독립적으로 latch됩니다. 예를 들어 R을 누른 뒤 W를 누르면 앞쪽은 든 상태로 몸체 curl을 계속 출력합니다.

macOS에서는 viewer 명령이 활성 conda 환경의 `mjpython`으로 자동 재실행됩니다. 따라서 사용자 명령은 그대로 `barisimulation manual ...`을 사용합니다.
