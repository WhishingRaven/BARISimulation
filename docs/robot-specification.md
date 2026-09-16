# 로봇 상세 사양

## 외형과 질량

| 항목 | 값 |
|---|---:|
| 전체 폭 | 0.10 m |
| 전체 길이 | 0.15 m |
| 전체 높이 | 0.01 m |
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
- 조향용 형상이나 yaw 관절은 추가하지 않습니다. `TURN_LEFT/RIGHT`는 한 discrete action 동안 자유 본체 전체의 yaw를 8° 바꾸는 조정된 동작입니다.

## 뒤쪽 가시

- 위치: 로봇 뒤쪽 끝의 밑면, 폭 0.10 m 전체에 걸친 한 줄
- 허용 strain: 최대 50 g-force = 0.4905 N
- 대상: 환경 표면 또는 다른 로봇
- 동작: 가시 줄이 접촉 중일 때만 `ATTACH` 가능하며, 허용 하중을 넘으면 constraint가 해제됩니다.
- 구현: 가시는 별도 충돌/시각 형상이 아닙니다. 기존 rear body contact 위의 비가시 attachment site만 사용합니다.

## 초음파 센서

| observation | 위치 | 방향 |
|---|---|---|
| `distance1` | 앞쪽 끝 가운데 | 전방 |
| `distance2` | 앞끝에서 0.04 m 뒤, 밑면 | 아래 |
| `distance3` | 앞끝에서 0.04 m 뒤, 왼쪽 면 | 왼쪽 |
| `distance4` | 앞끝에서 0.04 m 뒤, 오른쪽 면 | 오른쪽 |

센서는 비가시 ray로 계산하며 별도 충돌 형상이 없습니다. 현재 ray 상한은 1.0 m이고 검출이 없으면 1.0 m를 반환합니다.

## 시뮬레이터 조정값

고정 외형 사양을 바꾸지 않는 물리 제어값은 action 구현에 포함됩니다.

- policy/control interval: 0.5 s
- MuJoCo physics timestep: 0.002 s
- curl target: 52°
- front lift target: 52° 위쪽
- 한 turn action: 본체 전체 yaw 8°
- 로컬 ID 통신 반경: 0.50 m

통신 반경과 센서 최대 거리는 원 사양에 값이 없어서 명시적으로 둔 시뮬레이터 파라미터입니다.
