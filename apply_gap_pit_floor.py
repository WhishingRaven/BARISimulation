"""틈(gap) 아래 0.5 m에 받침 바닥을 깔아, 떨어진 로봇이 끝없이 추락하지 않게 한다.

git apply가 실패할 때 대신 쓰는 스크립트.
사용: BARISimulation 폴더에서  python apply_gap_pit_floor.py
되돌리기: git checkout -- bari_sim/simulation/scene.py bari_sim/tasks/evaluation.py
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
scene = root / "bari_sim" / "simulation" / "scene.py"
evaluation = root / "bari_sim" / "tasks" / "evaluation.py"
ok = True

# 1) scene.py: 깊이 상수 + 받침 바닥 geom
s = scene.read_text(encoding="utf-8")
if "environment_gap_pit" in s:
    print("scene.py      : 이미 적용됨")
else:
    anchor = 'SEGMENT_BRIGHTNESS = {"rear": 0.78, "middle": 1.0, "front": 1.18}'
    old = """                        **common,
                    },
                )
        else:"""
    new = """                        **common,
                    },
                )
            # 틈 아래 받침 바닥: 떨어진 로봇이 끝없이 추락하는 것을 막는다.
            pit = dict(common)
            pit.update(
                {
                    "name": "environment_gap_pit",
                    "type": "box",
                    "pos": _numbers((0.0, 0.0, -GAP_PIT_DEPTH_M - thickness / 2.0)),
                    "size": _numbers(
                        (gap_width / 2.0, platform_width / 2.0, thickness / 2.0)
                    ),
                    "rgba": "0.30 0.33 0.36 1",
                }
            )
            ET.SubElement(world, "geom", pit)
        else:"""
    if s.count(anchor) == 1 and s.count(old) == 1:
        s = s.replace(anchor, anchor + "\nGAP_PIT_DEPTH_M = 0.5   # 틈 아래 받침 바닥 깊이 (m)", 1)
        s = s.replace(old, new, 1)
        scene.write_text(s, encoding="utf-8")
        print("scene.py      : 적용 완료")
    else:
        ok = False
        print(f"scene.py      : 실패 — 기준 위치를 찾지 못함 "
              f"(상수 {s.count(anchor)}곳, gap 분기 {s.count(old)}곳, 각각 1곳이어야 함)")

# 2) evaluation.py: gap 성공 판정에 높이 조건 추가
e = evaluation.read_text(encoding="utf-8")
if "and z >= -self.robot.height_m\n            )" in e and "gap_width_m / 2.0 + self.robot.length_m / 2.0\n" in e:
    print("evaluation.py : 이미 적용됨")
else:
    old2 = "            return x >= self.scene.gap_width_m / 2.0 + self.robot.length_m / 2.0"
    new2 = ("            # 받침 바닥 때문에 떨어진 로봇이 건너편 플랫폼 아래에 있을 수 있으므로\n"
            "            # 플랫폼 높이에 있는지도 확인한다.\n"
            "            return (\n"
            "                x >= self.scene.gap_width_m / 2.0 + self.robot.length_m / 2.0\n"
            "                and z >= -self.robot.height_m\n"
            "            )")
    if e.count(old2) == 1:
        evaluation.write_text(e.replace(old2, new2, 1), encoding="utf-8")
        print("evaluation.py : 적용 완료")
    else:
        ok = False
        print(f"evaluation.py : 실패 — gap 성공 판정을 {e.count(old2)}곳에서 찾음 (1곳이어야 함)")

sys.exit(0 if ok else 1)
