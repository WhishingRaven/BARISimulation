# Action과 observation

## Observation

개별 policy가 받는 값은 아래 항목뿐입니다. 전역 위치, 전역 방향, 목표 좌표, 지도, 다른 로봇의 action은 포함하지 않습니다.

| Observation | 의미 | Policy feature | LinearPolicy 입력값 계산 |
|---|---|---|---|
| `distance1` | 앞쪽 거리 | `front_proximity` | \(1-\operatorname{clip}(distance1,0,R)/R\) |
| `distance2` | 아래쪽 거리 | `down_proximity` | \(1-\operatorname{clip}(distance2,0,R)/R\) |
| `distance3` | 왼쪽 거리 | `left_proximity` | \(1-\operatorname{clip}(distance3,0,R)/R\) |
| `distance4` | 오른쪽 거리 | `right_proximity` | \(1-\operatorname{clip}(distance4,0,R)/R\) |
| `nearby_robot_ids` | 통신 범위 안의 로봇들 | `nearby_robot_fraction` | \(\min(\lvert nearby\_robot\_ids\rvert/29,1)\) |
| `strain_value` | attachment 장력 | `strain_fraction` | \(\operatorname{clip}(strain\_value/G_{max},0,1)\) |
| `is_curled` | 몸체가 말린 상태인지 | `is_curled` | `float(is_curled)` → 0 또는 1 |
| `is_front_lifted` | 앞부분이 들렸는지 | `is_front_lifted` | `float(is_front_lifted)` → 0 또는 1 |
| `is_possible_to_attach` | attachment 가능 여부 | `is_possible_to_attach` | `float(is_possible_to_attach)` → 0 또는 1 |
| `is_attaching` | 현재 연결 중인지 | `is_attaching` | `float(is_attaching)` → 0 또는 1 |
| `is_detached` | 최근 분리되었는지 | `is_detached` | `float(is_detached)` → 0 또는 1 |
| *(관측값 없음)* | 선형 모델의 상수항 | `bias` | 항상 `1.0` |

여기서 `R`은 로봇 센서 최대 거리(`sensor_range_m`), `G_max`는 최대 strain 값(`maximum_strain_g`)입니다. 따라서 LinearPolicy에 들어가는 feature 벡터의 순서는 다음과 같습니다.

\[
\mathbf{x} =
\begin{bmatrix}
1,
front\_proximity,
down\_proximity,
left\_proximity,
right\_proximity,
nearby\_robot\_fraction,
strain\_fraction,
is\_curled,
is\_front\_lifted,
is\_possible\_to\_attach,
is\_attaching,
is\_detached
\end{bmatrix}^{T}
\]

- `nearby_robot_ids`: 매 step 통신 반경 0.50 m 이내 ID를 정렬해 저장합니다.
- `strain_value`: 현재 가시 constraint 하중을 gram-force로 변환한 값입니다. 최대 표시값은 100입니다.
- `is_possible_to_attach`: 현재 rear spike line이 적합한 환경/로봇 표면에 접촉했거나 attachment tolerance 이내로 접근해 있습니다.
- `is_attaching`: attachment constraint가 현재 활성 상태입니다.
- `is_detached`: 다른 로봇과 연결되었거나 다른 로봇 접촉 중 overload로 해제된 step에 참입니다.

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
| `TURN_LEFT` | A | 현재 자세를 유지하고 5° heading 목표를 향해 yaw-rate feedback motor torque로 반시계 방향 회전 |
| `TURN_RIGHT` | D | 현재 자세를 유지하고 5° heading 목표를 향해 yaw-rate feedback motor torque로 시계 방향 회전 |
| `STOP` | C | 현재 rear 관절 자세 유지 |

현재 `W_n`의 curl 강도 선택지는 구현하지 않고 하나의 `CURL_BODY`만 제공합니다.
W에서는 앞쪽 cleat, S에서는 뒤쪽 cleat가 바닥 또는 아래 로봇 상판에 물리 latch되어 교대로 접지합니다. A/D는 이 gait cleat를 활성화하지 않습니다.
W와 S를 번갈아 선택하면 이 교대 접지와 관절 운동으로 +x 전방 이동이 발생합니다.
명시적으로 `ATTACH`를 유지하면 기존 뒤쪽 spike attachment가 우선하며, 자동 cleat latch는 비활성화됩니다.
Manual 화면의 strain 라벨은 마지막 latch 위치에 계속 남으며, 해제 뒤에는 `0.0 g`로 갱신됩니다.

### `a_lift`

| policy 값 | manual 키 | 의미 |
|---|---|---|
| `LIFT_FRONT` | R | 앞쪽 0.02 m flap을 위로 듦 |
| `UNLIFT_FRONT` | F | 앞쪽 flap을 원래 자세로 내림 |
| `STOP` | - | 현재 앞쪽 flap 각도를 유지 |

### `a_grip`

| policy 값 | manual 키 | 의미 |
|---|---|---|
| `ATTACH` | Space | 뒤쪽 가시 접촉에 attachment 요청/유지 |
| `DETACH` | X | attachment 해제; 미부착 기본 출력 |
| `STOP` | - | 현재 attachment 상태를 유지 |

Policy는 세 component를 매 step 다시 출력합니다. Manual의 W/S는 키를 누르는 동안 반복 출력되고, 키를 놓으면 현재 0.5초 step 뒤 정지합니다. A/D는 한 번의 keydown으로 같은 5° 목표를 정착할 때까지 반복한 뒤 자동 정지합니다. lift와 grip component는 다음 입력까지 유지됩니다.
