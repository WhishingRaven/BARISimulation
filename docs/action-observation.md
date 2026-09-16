# Action과 observation

## Action

각 로봇은 0.5초마다 다음 tuple 하나를 받습니다.

```text
a = (a_motion, a_lift, a_grip)
```

### `a_motion`

| policy 값 | manual 키 | 의미 |
|---|---|---|
| `CURL_BODY` | W | 뒤쪽 관절을 접어 `^` 형상 생성 |
| `FLATTEN_BODY` | S | 뒤쪽 관절을 펴 `_` 형상 생성 |
| `TURN_LEFT` | A | 본체 전체를 반시계 방향으로 8° 회전 |
| `TURN_RIGHT` | D | 본체 전체를 시계 방향으로 8° 회전 |
| `STOP` | C | 현재 rear 관절 자세 유지 |

현재 `W_n`의 curl 강도 선택지는 구현하지 않고 하나의 `CURL_BODY`만 제공합니다. W와 S를 번갈아 선택하면 접촉 마찰로 +x 전방 이동이 발생합니다.

### `a_lift`

| policy 값 | manual 키 | 의미 |
|---|---|---|
| `LIFT_FRONT` | R | 앞쪽 0.02 m flap을 위로 듦 |
| `UNLIFT_FRONT` | F | 앞쪽 flap을 원래 자세로 내림 |

### `a_grip`

| policy 값 | manual 키 | 의미 |
|---|---|---|
| `ATTACH` | Space | 뒤쪽 가시 접촉에 attachment 요청/유지 |
| `DETACH` | X | attachment 해제; 미부착 기본 출력 |

Policy는 세 component를 매 step 다시 출력합니다. Manual 입력은 키를 한 번 누르면 해당 component가 이후 step에도 유지됩니다.

## Observation

개별 policy가 받는 값은 아래 항목뿐입니다. 전역 위치, 전역 방향, 목표 좌표, 지도, 다른 로봇의 action은 포함하지 않습니다.

```text
nearby_robot_ids: tuple[int, ...]
strain_value: float                 # g-force, 0..50
distance1, distance2, distance3, distance4: float  # metre
is_curled: bool
is_front_lifted: bool
is_possible_to_attach: bool
is_attaching: bool
is_detached: bool
```

- `nearby_robot_ids`: 매 step 통신 반경 0.50 m 이내 ID를 정렬해 저장합니다.
- `strain_value`: 현재 가시 constraint 하중을 gram-force로 변환한 값입니다. 최대 표시값은 50입니다.
- `is_possible_to_attach`: 현재 rear spike line에 적합한 환경/로봇 contact가 있습니다.
- `is_attaching`: attachment constraint가 현재 활성 상태입니다.
- `is_detached`: 다른 로봇과 연결되었거나 다른 로봇 접촉 중 overload로 해제된 step에 참입니다.
