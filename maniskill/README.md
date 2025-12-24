# AlohaMini ManiSkill3 Integration

AlohaMini 듀얼 암 모바일 로봇을 ManiSkill3 시뮬레이션 환경에서 사용하기 위한 통합 가이드입니다.

## Overview

AlohaMini는 다음과 같은 구성을 가진 듀얼 암 모바일 로봇입니다:
- **모바일 베이스**: 3개의 옴니휠 (omnidirectional wheels)
- **수직 리프트**: 1 DOF 프리즘매틱 조인트
- **듀얼 암**: 좌/우 각 6 DOF 매니퓰레이터

**총 DOF**: 16 (바퀴 3 + 리프트 1 + 좌팔 6 + 우팔 6)

## Directory Structure

```
maniskill/
├── agents/aloha_mini/           # 에이전트 클래스 파일
│   ├── __init__.py
│   ├── aloha_mini.py            # AlohaMini, AlohaMiniFixed
│   └── aloha_mini_virtual.py    # AlohaMiniVirtual (가상 베이스)
├── assets/robots/aloha_mini/    # URDF 및 메시 파일
│   ├── aloha_mini.urdf
│   ├── aloha_mini_virtual_base.urdf
│   └── meshes/                  # STL 메시 파일들
├── scene_builder/replicacad/    # 수정된 씬 빌더
│   └── scene_builder.py
├── examples/                    # 예제 스크립트
│   ├── demo_virtual_base.py     # 가상 베이스 데모 (권장)
│   ├── demo_ee_keyboard.py      # EE 키보드 컨트롤
│   └── run_replicacad.py        # ReplicaCAD 환경 실행
├── install.py                   # 설치 스크립트
└── README.md
```

## Installation

### 1. ManiSkill3 설치

```bash
pip install mani-skill
```

### 2. ReplicaCAD 데이터셋 다운로드

```bash
python -m mani_skill.utils.download_asset ReplicaCAD
```

### 3. AlohaMini 설치 (자동)

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

## Robot Variants

| Variant | UID | 설명 | 사용 사례 |
|---------|-----|------|----------|
| **AlohaMini** | `aloha_mini` | 실제 바퀴 물리 | 바퀴 마찰 연구 |
| **AlohaMiniFixed** | `aloha_mini_fixed` | 고정 베이스 | 조작 작업만 |
| **AlohaMiniVirtual** | `aloha_mini_virtual` | 가상 모바일 베이스 **(권장)** | 네비게이션 + 조작 |

> **권장**: `aloha_mini_virtual`을 사용하세요. XLeRobot과 동일한 방식으로 prismatic X/Y + rotation 조인트를 사용하여 안정적인 이동이 가능합니다.

## Quick Start

### 가상 베이스 데모 (권장)

```bash
cd maniskill/examples
python demo_virtual_base.py --render
```

**컨트롤 (FPS Style)**:
- `A/D`: 전진/후진
- `W/S`: 좌/우 이동 (strafe)
- `Q/E`: 좌/우 회전
- `R/F`: 리프트 업/다운

### EE 키보드 컨트롤

```bash
python demo_ee_keyboard.py --render
```

### ReplicaCAD 환경

```bash
python run_replicacad.py --render --control keyboard
```

## Python API

```python
import gymnasium as gym
import mani_skill.envs

# 가상 베이스 로봇 사용 (권장)
env = gym.make(
    "ReplicaCAD_SceneManipulation-v1",
    robot_uids="aloha_mini_virtual",
    render_mode="human",
    sim_backend="gpu",
    control_mode="pd_joint_pos",
    sensor_configs=dict(shader_pack="rt-fast"),
    human_render_camera_configs=dict(shader_pack="rt-fast"),
    enable_shadow=True,
    max_episode_steps=None,  # 무한 에피소드
)

obs, info = env.reset(options=dict(reconfigure=True))

while True:
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

## Controllers

### AlohaMiniVirtual (권장)

| Mode | Action Space | 설명 |
|------|--------------|------|
| `pd_joint_pos` | `[base(3), lift(1), left_arm(6), right_arm(6)]` = 16 DOF | 베이스 속도 + 암 위치 |

- `base[0]`: X 속도 (전진/후진)
- `base[1]`: Y 속도 (좌/우)
- `base[2]`: 회전 속도

### AlohaMini (실제 바퀴)

| Mode | Action Space | 설명 |
|------|--------------|------|
| `mobile_pd_joint_pos` | `[wheels(3), lift(1), left_arm(6), right_arm(6)]` = 16 DOF | 바퀴 속도 제어 |
| `pd_joint_pos` | `[lift(1), left_arm(6), right_arm(6)]` = 13 DOF | 고정 베이스 |

## Shader Options

| Shader | 설명 | 성능 |
|--------|------|------|
| `default` | 기본 래스터라이저 | 빠름 |
| `rt-fast` | 빠른 레이트레이싱 | 중간 |
| `rt` | 고품질 레이트레이싱 | 느림 |

## Troubleshooting

### 바퀴가 굴러가지만 로봇이 안 움직임

`aloha_mini_virtual` 로봇을 사용하세요:
```python
robot_uids="aloha_mini_virtual"
```

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

## References

- [ManiSkill3 Documentation](https://maniskill.readthedocs.io/)
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) - 가상 베이스 구현 참고
- [ReplicaCAD Dataset](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/scenes.html)
