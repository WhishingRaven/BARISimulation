"""앞 힌지 감쇠를 MuJoCo 관절 damping(암시적)으로 옮기는 수정을 적용한다.

git apply가 실패할 때(저장소 버전이 조금 다를 때) 대신 쓰는 스크립트.
사용: BARISimulation 폴더에서  python apply_front_hinge_fix.py
되돌리기: git checkout -- bari_sim/simulation/scene.py bari_sim/simulation/engine.py
"""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
scene = root / "bari_sim" / "simulation" / "scene.py"
engine = root / "bari_sim" / "simulation" / "engine.py"
ok = True

# 1) scene.py: 힌지 damping = 0.004 + kd
s = scene.read_text(encoding="utf-8")
if "0.004 + self.robot.joint_kd_nms_rad" in s:
    print("scene.py : 이미 적용됨")
else:
    new, n = re.subn(r'"damping":\s*"0\.004",',
                     '"damping": f"{0.004 + self.robot.joint_kd_nms_rad:.10g}",  # kd를 암시적 감쇠로', s)
    if n == 1:
        scene.write_text(new, encoding="utf-8")
        print("scene.py : 적용 완료")
    else:
        ok = False
        print(f'scene.py : 실패 — "damping": "0.004" 를 {n}곳에서 찾음 (1곳이어야 함)')

# 2) engine.py: Python 쪽 kd 토크 제거(회전 자세는 차이만)
e = engine.read_text(encoding="utf-8")
if "turn_joint_kd_nms_rad - self.robot.joint_kd_nms_rad" in e:
    print("engine.py: 이미 적용됨")
else:
    pat = re.compile(
        r"joint_kd\s*=\s*\(\s*self\.robot\.turn_joint_kd_nms_rad\s+if\s+turning\s+"
        r"else\s+self\.robot\.joint_kd_nms_rad\s*\)")
    rep = ("joint_kd = (  # kd는 MJCF 관절 damping으로 처리, 회전 자세만 차이를 명시적으로\n"
           "                self.robot.turn_joint_kd_nms_rad - self.robot.joint_kd_nms_rad\n"
           "                if turning\n"
           "                else 0.0\n"
           "            )")
    new, n = pat.subn(rep, e)
    if n == 1:
        engine.write_text(new, encoding="utf-8")
        print("engine.py: 적용 완료")
    else:
        ok = False
        print(f"engine.py: 실패 — joint_kd 계산 부분을 {n}곳에서 찾음 (1곳이어야 함). "
              "engine.py의 해당 부분을 복사해 보내 주세요.")

sys.exit(0 if ok else 1)
