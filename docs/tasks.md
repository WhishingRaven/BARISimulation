# Task와 난이도

공통 로봇 배치는 `2*5`, `4*5`, `6*5`입니다. 표기의 앞 수는 진행 방향의 row 수, 뒤 수는 가로 column 수입니다.

## 1. collision-avoidance

평평한 바닥에서 전체 로봇이 시작 군집을 유지하며 +x 방향 목표 지점으로 이동합니다.

| difficulty | 시작 centroid부터 목표까지 거리 |
|---:|---:|
| 1 | 2 m |
| 2 | 4 m |
| 3 | 6 m |
| 4 | 8 m |
| 5 | 10 m |

성공은 모든 로봇이 목표 x 기준 ±0.35 m 안에 도달한 경우입니다. 평가에는 다음 값을 기록합니다.

- 평균 위치 기준 거리 분산 `position_variance_m2`
- trajectory 전체 평균 위치 분산 `mean_position_variance_m2`
- 성공까지 걸린 `travel_time_s`
- 지속 접촉을 중복 계산하지 않은 로봇 쌍 충돌 event 수 `collision_count`
- 뒤집혀 이동 불능이 된 로봇 수 `flipped_immobile_robot_count`
- 목표 방향 진행률과 남은 거리

## 2. gap

로봇이 서로 협동해 두 platform 사이 gap을 건넙니다.

| difficulty | gap 폭 |
|---:|---:|
| 1 | 0.10 m |
| 2 | 0.15 m |
| 3 | 0.20 m |
| 4 | 0.25 m |
| 5 | 0.30 m |

전체 로봇의 80% 이상이 far platform으로 완전히 넘어가면 성공합니다. 일부 로봇 탈락을 허용하되 성공/탈락 수, 충돌, 뒤집힘을 모두 기록합니다.

## 3. step

로봇이 서로 협동해 +x 방향의 단차 platform 위로 올라갑니다.

| difficulty | step 높이 |
|---:|---:|
| 1 | 0.03 m |
| 2 | 0.05 m |
| 3 | 0.08 m |
| 4 | 0.10 m |
| 5 | 0.15 m |

전체 로봇의 80% 이상이 단차 상면에 올라가면 성공합니다. gap과 동일하게 일부 탈락을 허용합니다.

Task 성공 판정과 metric은 평가용 전역 상태를 사용하지만 이 정보는 개별 policy observation으로 전달되지 않습니다.
