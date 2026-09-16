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
- 조향용 형상이나 yaw 관절은 추가하지 않습니다. `TURN_LEFT/RIGHT`는 한 discrete action 동안 자유 본체 전체의 yaw를 8° 바꾸는 조정된 동작입니다.

## 교대 접지 cleat

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
