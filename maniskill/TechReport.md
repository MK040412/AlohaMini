# AlohaMini ManiSkill3 Integration - Technical Report

## 1. Overview

AlohaMini 듀얼 암 모바일 로봇을 ManiSkill3 시뮬레이션 환경에 통합하는 과정에서 발생한 문제와 해결 방안을 기술합니다.

---

## 2. System Architecture

### 2.1 Overall Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AlohaMini ManiSkill3 Integration                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   User Input    │────▶│   Controller    │────▶│    Physics      │       │
│  │   (Keyboard)    │     │   (PDBaseVel)   │     │    (SAPIEN)     │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│          │                       │                       │                  │
│          │                       │                       │                  │
│          ▼                       ▼                       ▼                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  FPS Controls   │     │  Action Space   │     │  Robot State    │       │
│  │  W/S/A/D/Q/E    │     │  [vx,vy,ω,...]  │     │  [qpos, qvel]   │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Codebase Structure

```
AlohaMini/
├── maniskill/                          # ManiSkill3 Integration Package
│   ├── agents/aloha_mini/              # Robot Agent Classes
│   │   ├── __init__.py                 # Exports: AlohaMini, AlohaMiniFixed, AlohaMiniVirtual
│   │   ├── aloha_mini.py               # Physical wheel + Fixed base variants
│   │   └── aloha_mini_virtual.py       # Virtual mobile base variant (Recommended)
│   │
│   ├── assets/robots/aloha_mini/       # Robot Assets
│   │   ├── aloha_mini.urdf             # Physical wheel URDF
│   │   ├── aloha_mini_virtual_base.urdf # Virtual base URDF (Recommended)
│   │   └── meshes/                     # STL mesh files (16 files)
│   │       ├── base_link.STL
│   │       ├── left_*.STL              # Left arm links (6)
│   │       ├── right_*.STL             # Right arm links (6)
│   │       ├── vertical_link.STL       # Lift link
│   │       └── wheel*.STL              # Wheel meshes (3)
│   │
│   ├── scene_builder/replicacad/       # Modified Scene Builder
│   │   └── scene_builder.py            # ReplicaCAD + AlohaMini support
│   │
│   ├── examples/                       # Demo Scripts
│   │   ├── demo_virtual_base.py        # Virtual base demo (Recommended)
│   │   ├── demo_ee_keyboard.py         # End-effector IK control
│   │   ├── run_replicacad.py           # ReplicaCAD environment demo
│   │   └── test_robot.py               # Basic robot test
│   │
│   ├── install.py                      # Auto-installer script
│   ├── README.md                       # Usage documentation
│   └── TechReport.md                   # This file
│
└── README.md                           # Main project README
```

---

## 3. Problem Statement

### 3.1 Initial Issue: Wheels Spinning Without Movement

```
┌──────────────────────────────────────────────────────────────────┐
│                    PROBLEM: Wheel Physics Failure                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Motor Torque ──▶ Wheel Rotation ──▶ Ground Friction ──X      │
│                                              │                   │
│                                              ▼                   │
│                                    ┌─────────────────┐          │
│                                    │  Robot Stays    │          │
│                                    │  Stationary!    │          │
│                                    └─────────────────┘          │
│                                                                  │
│    Root Cause: SAPIEN physics engine의 바퀴-지면 마찰 불안정      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**시도한 해결책들:**

| Attempt | Change | Result |
|---------|--------|--------|
| 1 | wheel_friction_material 추가 | ❌ 부분 개선 |
| 2 | wheel_force_limit: 100 → 500 | ❌ 부분 개선 |
| 3 | wheel_damping: 1000 → 50 | ❌ 부분 개선 |
| 4 | velocity bounds: ±1.0 → ±10.0 | ❌ 부분 개선 |

### 3.2 Analysis: XLeRobot Approach

[XLeRobot](https://github.com/Vector-Wangel/XLeRobot) 분석:

```
┌──────────────────────────────────────────────────────────────────┐
│                    DISCOVERY: XLeRobot Solution                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│    XLeRobot DOES NOT use actual wheel physics!                   │
│                                                                  │
│    Instead: Virtual Mobile Base with Prismatic Joints            │
│                                                                  │
│    ┌─────────────────────────────────────────────┐              │
│    │  root_x_axis_joint (prismatic) ──▶ X 이동   │              │
│    │  root_y_axis_joint (prismatic) ──▶ Y 이동   │              │
│    │  root_z_rotation_joint (continuous) ▶ 회전  │              │
│    └─────────────────────────────────────────────┘              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Design Principle

### 4.1 Virtual Mobile Base Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DESIGN DECISION: Virtual Base                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ❌ REAL WHEEL PHYSICS              ✅ VIRTUAL MOBILE BASE                 │
│   ┌───────────────────────┐          ┌───────────────────────┐             │
│   │                       │          │                       │             │
│   │  Motor Torque         │          │  Velocity Command     │             │
│   │       │               │          │       │               │             │
│   │       ▼               │          │       ▼               │             │
│   │  Wheel Rotation       │          │  Direct Position      │             │
│   │       │               │          │  Update               │             │
│   │       ▼               │    ▶▶    │       │               │             │
│   │  Ground Friction      │          │       ▼               │             │
│   │       │               │          │  Robot Moves!         │             │
│   │       ▼               │          │                       │             │
│   │  Robot Movement       │          └───────────────────────┘             │
│   │  (Unreliable!)        │                                                │
│   │                       │          ✅ Stable                             │
│   └───────────────────────┘          ✅ Predictable                        │
│                                      ✅ Same as XLeRobot                   │
│   ❌ Unstable                        ✅ Focus on manipulation              │
│   ❌ Slip issues                                                           │
│   ❌ Complex tuning                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Kinematic Chain Design

```
                              URDF Joint Hierarchy
                              ═══════════════════

world (fixed)
    │
    └──▶ root (dummy link, mass=0.0001)
           │
           └──▶ root_x_axis_joint ─────────────────────┐
                  │ type: prismatic                    │
                  │ axis: [1, 0, 0]                    │ X Translation
                  │ limit: [-20, +20] m                │
                  ▼                                    │
              root_x_link ─────────────────────────────┘
                  │
                  └──▶ root_y_axis_joint ──────────────┐
                         │ type: prismatic             │
                         │ axis: [0, 1, 0]             │ Y Translation
                         │ limit: [-20, +20] m         │
                         ▼                             │
                     root_y_link ──────────────────────┘
                         │
                         └──▶ root_z_rotation_joint ───┐
                                │ type: continuous     │
                                │ axis: [0, 0, 1]      │ Z Rotation
                                ▼                      │
                            base_link ─────────────────┘
                                │
                 ┌──────────────┼──────────────┬────────────────┐
                 │              │              │                │
                 ▼              ▼              ▼                ▼
           vertical_joint   wheel_1      left_joint_1    right_joint_1
           (prismatic)      (fixed)      (revolute)      (revolute)
                │              │              │                │
                ▼              ▼              ▼                ▼
          vertical_link   wheel_1_link  left_link1       right_link1
                               │              │                │
                               ▼              ▼                ▼
                          wheel_2,3      left_arm...      right_arm...
                          (fixed)
```

### 4.3 Action Space Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACTION SPACE (16 DOF)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Index   Joint                  Type          Range           Control      │
│   ─────   ─────                  ────          ─────           ───────      │
│     0     root_x_axis            velocity      [-1, +1] m/s    Base vx      │
│     1     root_y_axis            velocity      [-1, +1] m/s    Base vy      │
│     2     root_z_rotation        velocity      [-π, +π] rad/s  Base ω       │
│   ─────   ─────────────          ────────      ──────────      ────────     │
│     3     vertical_joint         position      [0, 0.6] m      Lift         │
│   ─────   ─────────────          ────────      ──────────      ────────     │
│    4-9    left_joint_1~6         position      varies          Left Arm     │
│   ─────   ─────────────          ────────      ──────────      ────────     │
│   10-15   right_joint_1~6        position      varies          Right Arm    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ action = [vx, vy, ω, lift, L1, L2, L3, L4, L5, L6, R1...R6]         │  │
│   │           ▲   ▲  ▲    ▲     └──────────────────────────────┘        │  │
│   │           │   │  │    │              Arm Position Control           │  │
│   │           └───┴──┴────┴─ Base Velocity + Lift Position              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Keyboard Control (FPS Style)

### 5.1 Control Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KEYBOARD CONTROLS (FPS Style)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────┐                                                  │
│                    │  W  │  Forward                                         │
│                    │ ▲▲▲ │                                                  │
│              ┌─────┼─────┼─────┐                                           │
│   Strafe    │  A  │     │  D  │  Strafe                                    │
│    Left     │ ◀── │     │ ──▶ │   Right                                    │
│              └─────┼─────┼─────┘                                           │
│                    │  S  │  Backward                                        │
│                    │ ▼▼▼ │                                                  │
│                    └─────┘                                                  │
│                                                                             │
│   ┌─────┐         ┌─────┐                                                  │
│   │  Q  │ Rotate  │  E  │ Rotate                                           │
│   │ ◀◀◀ │ Left    │ ▶▶▶ │ Right                                            │
│   └─────┘         └─────┘                                                  │
│                                                                             │
│   ┌─────┐         ┌─────┐                                                  │
│   │  R  │ Lift    │  F  │ Lift                                             │
│   │ ▲▲▲ │ Up      │ ▼▼▼ │ Down                                             │
│   └─────┘         └─────┘                                                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  X = Reset All Positions    ESC = Quit                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 FPS Style Rationale

```
FPS 스타일 컨트롤:

    W = Forward      ──▶  전진
    S = Backward     ──▶  후진
    A = Strafe Left  ──▶  좌측 이동 (평행 이동)
    D = Strafe Right ──▶  우측 이동 (평행 이동)
    Q = Rotate Left  ──▶  좌회전
    E = Rotate Right ──▶  우회전

이점:
    ✅ 표준 FPS 게임 레이아웃
    ✅ 직관적인 방향 감각
    ✅ 한 손으로 모든 이동 제어 가능
    ✅ WASD로 이동, QE로 회전 - 명확한 구분
```

---

## 6. Robot Variants Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROBOT VARIANTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐      │
│   │   aloha_mini      │  │ aloha_mini_fixed  │  │aloha_mini_virtual │      │
│   ├───────────────────┤  ├───────────────────┤  ├───────────────────┤      │
│   │                   │  │                   │  │                   │      │
│   │   ┌─────────┐     │  │   ┌─────────┐     │  │   ┌─────────┐     │      │
│   │   │  ◯   ◯  │     │  │   │  ◯   ◯  │     │  │   │  ◯   ◯  │     │      │
│   │   │    ▼    │     │  │   │    ▼    │     │  │   │    ▼    │     │      │
│   │   │ ╔═════╗ │     │  │   │ ╔═════╗ │     │  │   │ ╔═════╗ │     │      │
│   │   │ ║     ║ │     │  │   │ ║     ║ │     │  │   │ ║     ║ │     │      │
│   │   │ ╚═════╝ │     │  │   │ ╚═════╝ │     │  │   │ ╚═════╝ │     │      │
│   │   │  ⚙ ⚙ ⚙  │     │  │   │  ■ ■ ■  │     │  │   │  ○ ○ ○  │     │      │
│   │   └─────────┘     │  │   └─────────┘     │  │   └─────────┘     │      │
│   │                   │  │                   │  │                   │      │
│   │ Base: Real Wheels │  │ Base: Fixed       │  │ Base: Virtual     │      │
│   │ DOF: 16           │  │ DOF: 13           │  │ DOF: 16           │      │
│   │                   │  │                   │  │                   │      │
│   │ ❌ Unstable       │  │ ✅ Stable         │  │ ✅ Stable         │      │
│   │ ❌ Slip issues    │  │ ❌ No mobility    │  │ ✅ Mobile         │      │
│   │                   │  │                   │  │ ✅ RECOMMENDED    │      │
│   └───────────────────┘  └───────────────────┘  └───────────────────┘      │
│                                                                             │
│   Use Case:              Use Case:              Use Case:                   │
│   Wheel dynamics         Manipulation only      Navigation +                │
│   research               (tabletop tasks)       Manipulation                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Installation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INSTALLATION PROCESS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User                                                                      │
│     │                                                                       │
│     │  $ python install.py                                                  │
│     ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        install.py                                   │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │                                                                     │  │
│   │  1. find_maniskill_path()                                           │  │
│   │     │                                                               │  │
│   │     │  import mani_skill                                            │  │
│   │     │  path = Path(mani_skill.__file__).parent                      │  │
│   │     ▼                                                               │  │
│   │     /path/to/site-packages/mani_skill/                              │  │
│   │                                                                     │  │
│   │  2. Copy Agent Files                                                │  │
│   │     agents/aloha_mini/*.py  ──▶  mani_skill/agents/robots/aloha_mini/│  │
│   │                                                                     │  │
│   │  3. Copy URDF & Meshes                                              │  │
│   │     assets/robots/aloha_mini/  ──▶  ~/.maniskill/data/robots/aloha_mini/│
│   │                                                                     │  │
│   │  4. Update Scene Builder                                            │  │
│   │     scene_builder.py  ──▶  mani_skill/utils/scene_builder/replicacad/│  │
│   │     (with backup)                                                   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│     │                                                                       │
│     ▼                                                                       │
│   ✅ Installation Complete!                                                 │
│                                                                             │
│   $ python demo_virtual_base.py --render                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Controller Architecture

### 8.1 PDBaseVelController

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PDBaseVelController Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Input (Ego-centric)              World Frame Output                  │
│   ┌───────────────────┐                 ┌───────────────────┐              │
│   │ vx = 0.5 (forward)│                 │ root_x velocity   │              │
│   │ vy = 0.0          │  ──Transform──▶ │ root_y velocity   │              │
│   │ ω  = 0.3 (rotate) │                 │ root_z velocity   │              │
│   └───────────────────┘                 └───────────────────┘              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   PDBaseVelController internally:                                   │  │
│   │                                                                     │  │
│   │   1. Get current robot orientation (θ)                              │  │
│   │   2. Transform ego-centric to world:                                │  │
│   │      world_vx = vx * cos(θ) - vy * sin(θ)                          │  │
│   │      world_vy = vx * sin(θ) + vy * cos(θ)                          │  │
│   │   3. Apply PD control to reach target velocity                      │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Controller Configuration

```python
# AlohaMiniVirtual Controller Config
base_pd_joint_vel = PDBaseVelControllerConfig(
    joint_names=["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"],
    lower=[-1, -1, -3.14],      # [vx_min, vy_min, omega_min]
    upper=[1, 1, 3.14],          # [vx_max, vy_max, omega_max]
    damping=1000,                # Velocity tracking damping
    force_limit=500,             # Maximum force
)
```

---

## 9. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐      │
│  │  pygame    │───▶│   Demo     │───▶│  Gymnasium │───▶│  ManiSkill │      │
│  │  Events    │    │  Script    │    │    Env     │    │   Agent    │      │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘      │
│       │                  │                  │                  │            │
│       │                  │                  │                  │            │
│       ▼                  ▼                  ▼                  ▼            │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐      │
│  │ Keypress   │    │  Action    │    │   step()   │    │ Controller │      │
│  │  W,S,A,D   │    │ [vx,vy,ω]  │    │            │    │ PDBaseVel  │      │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘      │
│                                             │                  │            │
│                                             │                  │            │
│                                             ▼                  ▼            │
│                                       ┌────────────┐    ┌────────────┐      │
│                                       │   SAPIEN   │◀───│   URDF     │      │
│                                       │  Physics   │    │  Joints    │      │
│                                       └────────────┘    └────────────┘      │
│                                             │                               │
│                                             │                               │
│                                             ▼                               │
│                                       ┌────────────┐                        │
│                                       │  Rendered  │                        │
│                                       │   Frame    │                        │
│                                       └────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Conclusion

### 10.1 Key Design Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Virtual Mobile Base | XLeRobot 방식 채택, 안정성 확보 | ✅ 안정적 이동 |
| 3 Robot Variants | 용도별 선택 가능 | ✅ 유연성 |
| PDBaseVelController | Ego-centric 속도 제어 | ✅ 직관적 제어 |
| FPS Style Controls | 게이머 친화적 | ✅ 익숙한 UX |
| Auto-installer | Zero Configuration | ✅ 쉬운 설치 |

### 10.2 Limitations & Future Work

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LIMITATIONS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ⚠️  바퀴 Sim-to-Real Gap: 가상 베이스는 실제 바퀴 동작과 다름              │
│  ⚠️  옴니휠 역학 미구현: 하드웨어의 4862 configuration 미반영               │
│  ⚠️  슬립/충돌 시뮬레이션 없음: 가상 베이스의 한계                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         FUTURE WORK                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📋 실제 옴니휠 역학 구현 (wheel velocity → base velocity)                  │
│  📋 그리퍼 제어 추가                                                        │
│  📋 카메라 센서 통합                                                        │
│  📋 RL 학습 환경 구성                                                       │
│  📋 Sim-to-Real transfer 검증                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## References

- [ManiSkill3 Documentation](https://maniskill.readthedocs.io/)
- [XLeRobot Repository](https://github.com/Vector-Wangel/XLeRobot)
- [SAPIEN Physics Engine](https://sapien.ucsd.edu/)
- [ManiSkill Fetch Robot](https://github.com/haosulab/ManiSkill/tree/main/mani_skill/agents/robots/fetch)
