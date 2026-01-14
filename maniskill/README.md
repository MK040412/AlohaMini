# AlohaMini ManiSkill3 Integration

AlohaMini 듀얼 암 모바일 로봇을 ManiSkill3 시뮬레이션 환경에서 사용하기 위한 통합 가이드입니다.

## Overview

AlohaMini는 다음과 같은 구성을 가진 듀얼 암 모바일 로봇입니다:
- **모바일 베이스**: 가상 prismatic X/Y + rotation 조인트
- **수직 리프트**: 1 DOF 프리즘매틱 조인트
- **듀얼 암**: 좌/우 각 6 DOF SO100 매니퓰레이터

**총 DOF**: 16 (베이스 3 + 리프트 1 + 좌팔 6 + 우팔 6)

## Directory Structure

```
maniskill/
├── agents/aloha_mini/           # 에이전트 클래스 파일
│   ├── __init__.py
│   ├── base_agent.py            # AlohaMiniBaseAgent (추상 클래스)
│   └── aloha_mini_so100_v2.py   # AlohaMiniSO100V2 (메인 에이전트)
├── assets/robots/aloha_mini/    # URDF 및 메시 파일
│   ├── maniskill_so100_version.urdf
│   └── so100_meshes/            # STL/PLY 메시 파일들
├── teleop/                      # 텔레오프레이션 모듈
│   ├── demo_teleop.py           # 키보드 IK 텔레오프 (권장)
│   ├── demo_vr_teleop_stream.py # VR 텔레오프 + 카메라 스트리밍
│   ├── controller.py            # TeleopController
│   ├── config.py                # TeleopConfig
│   ├── inputs/                  # 입력 핸들러 (keyboard, VR)
│   ├── kinematics/              # IK 모듈
│   └── web_ui_stream/           # VR 웹 UI
├── examples/                    # 예제 스크립트
│   ├── demo_ee_keyboard.py      # EE 키보드 컨트롤
│   └── run_replicacad.py        # ReplicaCAD 환경 실행
├── scene_builder/replicacad/    # 수정된 씬 빌더
│   └── scene_builder.py
├── install.py                   # 설치 스크립트
├── setup.py                     # 패키지 설정
└── README.md
```

## Installation

### 1. 가상환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. ManiSkill3 설치

```bash
pip install mani-skill
```

### 3. 추가 의존성 설치

```bash
pip install pygame websockets Pillow
```

### 4. AlohaMini 설치

```bash
cd maniskill
python install.py
```

이 스크립트는 자동으로:
- 에이전트 파일을 ManiSkill에 복사
- URDF/메시 파일을 `~/.maniskill/data/`에 복사
- ReplicaCAD 씬 빌더 업데이트

### 제거

```bash
python install.py --uninstall
```

## Robot Agent

| Agent | UID | 설명 |
|-------|-----|------|
| **AlohaMiniSO100V2** | `aloha_mini_so100_v2` | SO100 암을 사용하는 가상 베이스 로봇 |

> **참고**: 가상 베이스(prismatic X/Y + rotation)를 사용하여 안정적인 이동이 가능합니다.

## Quick Start

### 키보드 IK 텔레오프 (권장)

```bash
cd maniskill/teleop
python demo_teleop.py --render
```

**컨트롤 (XLeRobot Style)**:

| 왼팔 | 오른팔 | 기능 |
|------|--------|------|
| Q/A | U/J | Shoulder Pan -/+ |
| W/S | I/K | End-Effector X (전진/후진) |
| E/D | O/L | End-Effector Y (하강/상승) |
| R/F | P/; | Pitch -/+ |
| T/G | [/' | Wrist Roll -/+ |
| Y/H | ]/\ | Gripper 닫기/열기 |

| 일반 | 기능 |
|------|------|
| SPACE | 암 초기 위치로 리셋 |
| X/ESC | 종료 |

### VR 텔레오프 (카메라 스트리밍)

```bash
cd maniskill/teleop
python demo_vr_teleop_stream.py
```

VR 헤드셋 브라우저에서 `https://<your-ip>:8443` 접속

## Python API

```python
import gymnasium as gym
import mani_skill.envs

# Import agent to register
from mani_skill.agents.robots import aloha_mini

# Create environment
env = gym.make(
    "ReplicaCAD_SceneManipulation-v1",
    robot_uids="aloha_mini_so100_v2",
    render_mode="human",
    sim_backend="gpu",
    control_mode="pd_joint_pos",
    sensor_configs=dict(shader_pack="rt-fast"),
    human_render_camera_configs=dict(shader_pack="rt-fast"),
    enable_shadow=True,
)

obs, info = env.reset(options=dict(reconfigure=True))

while True:
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

## Controllers

### Action Space (pd_joint_pos)

| Index | Joint | 설명 |
|-------|-------|------|
| 0 | base_x | X 속도 (전진/후진) |
| 1 | base_y | Y 속도 (좌/우) |
| 2 | base_rot | 회전 속도 |
| 3 | lift | 리프트 위치 |
| 4-9 | left_arm | 왼팔 6 조인트 |
| 10-15 | right_arm | 오른팔 6 조인트 |

**총 16 DOF**

## Shader Options

| Shader | 설명 | 성능 |
|--------|------|------|
| `default` | 기본 래스터라이저 | 빠름 |
| `rt-fast` | 빠른 레이트레이싱 | 중간 |
| `rt` | 고품질 레이트레이싱 | 느림 |

## Troubleshooting

### 검은 화면

```python
env = gym.make(
    ...,
    sensor_configs=dict(shader_pack="default"),
    human_render_camera_configs=dict(shader_pack="default"),
    enable_shadow=True,
)
```

매 스텝마다 `env.render()` 호출 필수.

### 키보드 입력이 안됨

pygame 윈도우에 포커스를 맞추세요. 데모 스크립트는 자동으로 컨트롤 윈도우를 생성합니다.

### ManiSkill import 에러

`install.py`가 정상적으로 실행되었는지 확인하세요:
```bash
python install.py
```

## References

- [ManiSkill3 Documentation](https://maniskill.readthedocs.io/)
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) - 가상 베이스 구현 참고
- [ReplicaCAD Dataset](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/scenes.html)
