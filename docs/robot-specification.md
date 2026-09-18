# 로봇 상세 사양

## 외형과 질량

| 항목 | 값 |
|---|---:|
| 전체 폭 | 0.10 m |
| 전체 길이 | 0.15 m |
| 전체 높이 | 0.005 m |
| 전체 질량 | 0.05 kg (50 g) |

충돌 형상은 이 직육면체 외피를 길이 방향 세 segment로 나눈 것입니다.

```text
rear edge                                                    front edge
|<------ 0.06 m ------>|<---------- 0.07 m ---------->|<- 0.02 m ->|
          rear hinge ^                          front hinge ^
```

- 뒤쪽 0.06 m segment와 나머지 몸체 사이의 pitch 관절이 `CURL_BODY`를 수행합니다. 접으면 옆에서 본 몸체가 `^` 형상이 됩니다.
- 앞쪽 0.02 m flap은 별도 pitch 관절로만 움직이며 `LIFT_FRONT`에서 위로 들립니다.
- 질량은 세 segment 길이에 비례해 분배되고 합계는 항상 0.05 kg입니다.
- 조향은 접지 cleat 없이 root free joint의 yaw motor가 최대 \(\pm0.050\ \mathrm{N\cdot m}\) 토크를 가하는 방식입니다. 위치나 quaternion을 직접 수정하지 않으므로 회전은 MuJoCo dynamics가 풉니다. 각 5° heading 목표의 남은 오차로 부드러운 목표 yaw 속도를 만들고, 속도 오차에 비례한 torque로 가속·감속합니다. 목표에 늦게 도달하는 불안정 접지에서는 같은 목표를 유지하며, 도달한 뒤에만 다음 5° 단위로 진행합니다. W/S latch가 풀린 직후에는 pitch hinge를 낮은 토크의 compliance 제어로 유지해 접촉 변화가 떨림이나 수평 미끄러짐으로 증폭되지 않게 합니다.

## 교대 접지 cleat

각 cleat는 외형 끝단보다 3 mm 안쪽에 있습니다. 이 배치는 보행 중 ray가 하부 로봇 상판의 모서리를 빗나가지 않게 하며, W/S 동작의 latch 반작용 힘이 하부 로봇으로 전달되게 합니다. A/D 조향에는 cleat를 사용하지 않습니다.

앞·뒤 끝에는 폭 전체를 덮는 낮은 cleat가 하나씩 있습니다. `CURL_BODY`에서는
앞 cleat, `FLATTEN_BODY`에서는 뒤 cleat가 현재 바닥 또는 아래 로봇의 상판
접지점에 latch되어 몸통의 힌지 운동을 반대쪽 끝의 전진으로 바꿉니다. latch는 MuJoCo의 site-to-site
`connect` constraint로 표현하며, 로봇의 free-joint 위치나 속도를 직접 보정하지
않습니다. 다른 로봇 상판을 잡으면 constraint의 반작용 힘은 해당 아래 로봇의
link body에 전달됩니다. 따라서 남는 미세한 solver compliance는 실제 물리 결과로 유지됩니다.

## 뒤쪽 가시

- 위치: 로봇 뒤쪽 끝의 밑면, 폭 0.10 m 전체에 걸친 한 줄
- 허용 strain: 최대 100 g-force = 0.981 N
- 대상: 환경 표면 또는 다른 로봇
- 동작: 가시 줄이 접촉 중일 때만 `ATTACH` 가능하며, 허용 하중을 넘으면 constraint가 해제됩니다.
- 구현: 가시는 별도 충돌/시각 형상이 아닙니다. 기존 rear body contact 위의 비가시 attachment site만 사용합니다.

## 초음파 센서

| observation | 위치 | 방향 | 최대 거리 |
|---|---|---|---:|
| `distance1` | 앞쪽 끝 가운데 | 전방 | 1.00 m |
| `distance2` | 앞끝에서 0.04 m 뒤, 밑면 | 아래 | 1.00 m |
| `distance3` | 앞끝에서 0.04 m 뒤, 왼쪽 면 | 왼쪽 | 1.00 m |
| `distance4` | 앞끝에서 0.04 m 뒤, 오른쪽 면 | 오른쪽 | 1.00 m |

센서는 비가시 ray로 계산하며 별도 충돌 형상이 없습니다. 모든 센서의 최대 측정 거리는 `sensor_range_m = 1.00 m`입니다. 최대 거리 안에서 검출하지 못하면 1.00 m를 반환합니다.

## 로봇 간 통신

로봇은 다른 로봇의 중심 위치가 통신 범위 안에 있을 때 해당 로봇의 ID를 observation에 포함합니다.

| 항목 | 값 |
|---|---:|
| 통신 사거리 | `communication_range_m = 0.50 m` |
| observation 필드 | `nearby_robot_ids` |

통신 사거리 이내의 로봇만 `nearby_robot_ids`에 포함되며, 자기 자신은 포함되지 않습니다.

## 시뮬레이터 조정값

고정 외형 사양을 바꾸지 않는 물리 제어값은 action 구현에 포함됩니다.

생성 직후에는 바닥과의 작은 초기 간극을 0.5초 무제어 물리 정착으로 해소하고, 시간과 속도를 0으로 되돌린 뒤 첫 policy action을 받습니다. 따라서 첫 A/D·W/S 입력도 이후 입력과 같은 접지 조건에서 시작합니다.

- policy/control interval: 0.5 s
- MuJoCo physics timestep: 0.002 s
- curl target: 52°
- front lift target: 52° 위쪽
- turn yaw torque: 0.050 N·m, max yaw speed: 20°/s, increment: 5° (cleat 없이 yaw-rate feedback으로 회전)
- 센서 최대 측정 거리: 1.00 m
- 로컬 ID 통신 반경: 0.50 m

통신 반경과 센서 최대 거리는 원 사양에 값이 없어서 명시적으로 둔 시뮬레이터 파라미터입니다.
