// AlohaMini VR Teleoperation App (XLeRobot style)
// Auto-connects to WebSocket server when scene loads

AFRAME.registerComponent('controller-updater', {
  init: function () {
    console.log("Controller updater component initialized.");

    this.leftHand = document.querySelector('#leftHand');
    this.rightHand = document.querySelector('#rightHand');
    this.leftHandInfoText = document.querySelector('#leftHandInfo');
    this.rightHandInfoText = document.querySelector('#rightHandInfo');

    // Headset tracking
    this.headset = document.querySelector('#headset');
    this.headsetInfoText = document.querySelector('#headsetInfo');

    // WebSocket Setup
    this.websocket = null;
    this.leftGripDown = false;
    this.rightGripDown = false;
    this.leftTriggerDown = false;
    this.rightTriggerDown = false;

    // Relative rotation tracking
    this.leftGripInitialRotation = null;
    this.rightGripInitialRotation = null;
    this.leftRelativeRotation = { x: 0, y: 0, z: 0 };
    this.rightRelativeRotation = { x: 0, y: 0, z: 0 };

    // Quaternion-based Z-axis rotation tracking
    this.leftGripInitialQuaternion = null;
    this.rightGripInitialQuaternion = null;
    this.leftZAxisRotation = 0;
    this.rightZAxisRotation = 0;

    // WebSocket connection (XLeRobot style - auto-connect)
    const serverHostname = window.location.hostname;
    const websocketPort = 8442;
    const websocketUrl = `wss://${serverHostname}:${websocketPort}`;
    console.log(`Attempting WebSocket connection to: ${websocketUrl}`);

    try {
      this.websocket = new WebSocket(websocketUrl);
      this.websocket.onopen = (event) => {
        console.log(`WebSocket connected to ${websocketUrl}`);
        this.updateStatusIndicator('wsStatus', true);
      };
      this.websocket.onerror = (event) => {
        console.error(`WebSocket Error:`, event);
        this.updateStatusIndicator('wsStatus', false);
      };
      this.websocket.onclose = (event) => {
        console.log(`WebSocket disconnected. Clean: ${event.wasClean}, Code: ${event.code}`);
        this.websocket = null;
        this.updateStatusIndicator('wsStatus', false);
      };
      this.websocket.onmessage = (event) => {
        console.log(`WebSocket message: ${event.data}`);
      };
    } catch (error) {
      console.error(`Failed to create WebSocket:`, error);
      this.updateStatusIndicator('wsStatus', false);
    }

    if (!this.leftHand || !this.rightHand) {
      console.error("Controller entities not found!");
      return;
    }

    // Apply text rotation
    const textRotation = '-90 0 0';
    if (this.leftHandInfoText) this.leftHandInfoText.setAttribute('rotation', textRotation);
    if (this.rightHandInfoText) this.rightHandInfoText.setAttribute('rotation', textRotation);

    // Create axis indicators
    this.createAxisIndicators();

    // Helper: send grip release
    this.sendGripRelease = (hand) => {
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({ hand: hand, gripReleased: true }));
        console.log(`Sent grip release for ${hand} hand`);
      }
    };

    // Helper: send trigger release
    this.sendTriggerRelease = (hand) => {
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({ hand: hand, triggerReleased: true }));
        console.log(`Sent trigger release for ${hand} hand`);
      }
    };

    // Helper: calculate relative rotation
    this.calculateRelativeRotation = (current, initial) => {
      return {
        x: current.x - initial.x,
        y: current.y - initial.y,
        z: current.z - initial.z
      };
    };

    // Helper: calculate Z-axis rotation from quaternions
    this.calculateZAxisRotation = (currentQuat, initialQuat) => {
      const relativeQuat = new THREE.Quaternion();
      relativeQuat.multiplyQuaternions(currentQuat, initialQuat.clone().invert());

      const forwardDirection = new THREE.Vector3(0, 0, 1);
      forwardDirection.applyQuaternion(currentQuat);

      const angle = 2 * Math.acos(Math.abs(relativeQuat.w));
      if (angle < 0.0001) return 0;

      const sinHalfAngle = Math.sqrt(1 - relativeQuat.w * relativeQuat.w);
      const rotationAxis = new THREE.Vector3(
        relativeQuat.x / sinHalfAngle,
        relativeQuat.y / sinHalfAngle,
        relativeQuat.z / sinHalfAngle
      );

      const projectedComponent = rotationAxis.dot(forwardDirection);
      let degrees = THREE.MathUtils.radToDeg(angle * projectedComponent);

      while (degrees > 180) degrees -= 360;
      while (degrees < -180) degrees += 360;

      return degrees;
    };

    // Event listeners - Left hand
    this.leftHand.addEventListener('triggerdown', () => {
      console.log('Left Trigger Pressed');
      this.leftTriggerDown = true;
    });
    this.leftHand.addEventListener('triggerup', () => {
      console.log('Left Trigger Released');
      this.leftTriggerDown = false;
      this.sendTriggerRelease('left');
    });
    this.leftHand.addEventListener('gripdown', () => {
      console.log('Left Grip Pressed');
      this.leftGripDown = true;
      if (this.leftHand.object3D.visible) {
        const rot = this.leftHand.object3D.rotation;
        this.leftGripInitialRotation = {
          x: THREE.MathUtils.radToDeg(rot.x),
          y: THREE.MathUtils.radToDeg(rot.y),
          z: THREE.MathUtils.radToDeg(rot.z)
        };
        this.leftGripInitialQuaternion = this.leftHand.object3D.quaternion.clone();
      }
    });
    this.leftHand.addEventListener('gripup', () => {
      console.log('Left Grip Released');
      this.leftGripDown = false;
      this.leftGripInitialRotation = null;
      this.leftGripInitialQuaternion = null;
      this.leftRelativeRotation = { x: 0, y: 0, z: 0 };
      this.leftZAxisRotation = 0;
      this.sendGripRelease('left');
    });

    // Event listeners - Right hand
    this.rightHand.addEventListener('triggerdown', () => {
      console.log('Right Trigger Pressed');
      this.rightTriggerDown = true;
    });
    this.rightHand.addEventListener('triggerup', () => {
      console.log('Right Trigger Released');
      this.rightTriggerDown = false;
      this.sendTriggerRelease('right');
    });
    this.rightHand.addEventListener('gripdown', () => {
      console.log('Right Grip Pressed');
      this.rightGripDown = true;
      if (this.rightHand.object3D.visible) {
        const rot = this.rightHand.object3D.rotation;
        this.rightGripInitialRotation = {
          x: THREE.MathUtils.radToDeg(rot.x),
          y: THREE.MathUtils.radToDeg(rot.y),
          z: THREE.MathUtils.radToDeg(rot.z)
        };
        this.rightGripInitialQuaternion = this.rightHand.object3D.quaternion.clone();
      }
    });
    this.rightHand.addEventListener('gripup', () => {
      console.log('Right Grip Released');
      this.rightGripDown = false;
      this.rightGripInitialRotation = null;
      this.rightGripInitialQuaternion = null;
      this.rightRelativeRotation = { x: 0, y: 0, z: 0 };
      this.rightZAxisRotation = 0;
      this.sendGripRelease('right');
    });
  },

  updateStatusIndicator: function(id, connected) {
    const el = document.getElementById(id);
    if (el) {
      el.style.backgroundColor = connected ? '#00ff00' : '#ff0000';
    }
  },

  createAxisIndicators: function() {
    // Left Controller XYZ axes
    const leftX = document.createElement('a-cylinder');
    leftX.setAttribute('height', '0.08');
    leftX.setAttribute('radius', '0.003');
    leftX.setAttribute('color', '#ff0000');
    leftX.setAttribute('position', '0.04 0 0');
    leftX.setAttribute('rotation', '0 0 90');
    this.leftHand.appendChild(leftX);

    const leftY = document.createElement('a-cylinder');
    leftY.setAttribute('height', '0.08');
    leftY.setAttribute('radius', '0.003');
    leftY.setAttribute('color', '#00ff00');
    leftY.setAttribute('position', '0 0.04 0');
    this.leftHand.appendChild(leftY);

    const leftZ = document.createElement('a-cylinder');
    leftZ.setAttribute('height', '0.08');
    leftZ.setAttribute('radius', '0.003');
    leftZ.setAttribute('color', '#0000ff');
    leftZ.setAttribute('position', '0 0 0.04');
    leftZ.setAttribute('rotation', '90 0 0');
    this.leftHand.appendChild(leftZ);

    // Right Controller XYZ axes
    const rightX = document.createElement('a-cylinder');
    rightX.setAttribute('height', '0.08');
    rightX.setAttribute('radius', '0.003');
    rightX.setAttribute('color', '#ff0000');
    rightX.setAttribute('position', '0.04 0 0');
    rightX.setAttribute('rotation', '0 0 90');
    this.rightHand.appendChild(rightX);

    const rightY = document.createElement('a-cylinder');
    rightY.setAttribute('height', '0.08');
    rightY.setAttribute('radius', '0.003');
    rightY.setAttribute('color', '#00ff00');
    rightY.setAttribute('position', '0 0.04 0');
    this.rightHand.appendChild(rightY);

    const rightZ = document.createElement('a-cylinder');
    rightZ.setAttribute('height', '0.08');
    rightZ.setAttribute('radius', '0.003');
    rightZ.setAttribute('color', '#0000ff');
    rightZ.setAttribute('position', '0 0 0.04');
    rightZ.setAttribute('rotation', '90 0 0');
    this.rightHand.appendChild(rightZ);

    console.log('XYZ axis indicators created (RGB)');
  },

  tick: function () {
    if (!this.leftHand || !this.rightHand) return;

    const leftController = {
      hand: 'left',
      position: null,
      rotation: null,
      quaternion: null,
      gripActive: false,
      trigger: 0,
      thumbstick: { x: 0, y: 0 },
      buttons: {}
    };

    const rightController = {
      hand: 'right',
      position: null,
      rotation: null,
      quaternion: null,
      gripActive: false,
      trigger: 0,
      thumbstick: { x: 0, y: 0 },
      buttons: {}
    };

    const headset = {
      position: null,
      rotation: null,
      quaternion: null
    };

    // Left Hand
    if (this.leftHand && this.leftHand.object3D) {
      const pos = this.leftHand.object3D.position;
      const rot = this.leftHand.object3D.rotation;
      const rotDeg = {
        x: THREE.MathUtils.radToDeg(rot.x),
        y: THREE.MathUtils.radToDeg(rot.y),
        z: THREE.MathUtils.radToDeg(rot.z)
      };

      if (this.leftGripDown && this.leftGripInitialRotation) {
        this.leftRelativeRotation = this.calculateRelativeRotation(rotDeg, this.leftGripInitialRotation);
        if (this.leftGripInitialQuaternion) {
          this.leftZAxisRotation = this.calculateZAxisRotation(
            this.leftHand.object3D.quaternion,
            this.leftGripInitialQuaternion
          );
        }
      }

      let text = `Pos: ${pos.x.toFixed(2)} ${pos.y.toFixed(2)} ${pos.z.toFixed(2)}\nRot: ${rotDeg.x.toFixed(0)} ${rotDeg.y.toFixed(0)} ${rotDeg.z.toFixed(0)}`;
      if (this.leftGripDown) text += `\nZ-Rot: ${this.leftZAxisRotation.toFixed(1)}`;
      if (this.leftHandInfoText) this.leftHandInfoText.setAttribute('value', text);

      leftController.position = { x: pos.x, y: pos.y, z: pos.z };
      leftController.rotation = rotDeg;
      leftController.quaternion = {
        x: this.leftHand.object3D.quaternion.x,
        y: this.leftHand.object3D.quaternion.y,
        z: this.leftHand.object3D.quaternion.z,
        w: this.leftHand.object3D.quaternion.w
      };
      leftController.trigger = this.leftTriggerDown ? 1 : 0;
      leftController.gripActive = this.leftGripDown;

      // Get thumbstick from tracked-controls
      if (this.leftHand.components && this.leftHand.components['tracked-controls']) {
        const gamepad = this.leftHand.components['tracked-controls'].controller?.gamepad;
        if (gamepad) {
          leftController.thumbstick = {
            x: gamepad.axes[2] || 0,
            y: gamepad.axes[3] || 0
          };
          leftController.buttons = {
            a: !!gamepad.buttons[3]?.pressed,
            b: !!gamepad.buttons[4]?.pressed,
            squeeze: !!gamepad.buttons[1]?.pressed,
            thumbstick: !!gamepad.buttons[2]?.pressed
          };
        }
      }
    }

    // Right Hand
    if (this.rightHand && this.rightHand.object3D) {
      const pos = this.rightHand.object3D.position;
      const rot = this.rightHand.object3D.rotation;
      const rotDeg = {
        x: THREE.MathUtils.radToDeg(rot.x),
        y: THREE.MathUtils.radToDeg(rot.y),
        z: THREE.MathUtils.radToDeg(rot.z)
      };

      if (this.rightGripDown && this.rightGripInitialRotation) {
        this.rightRelativeRotation = this.calculateRelativeRotation(rotDeg, this.rightGripInitialRotation);
        if (this.rightGripInitialQuaternion) {
          this.rightZAxisRotation = this.calculateZAxisRotation(
            this.rightHand.object3D.quaternion,
            this.rightGripInitialQuaternion
          );
        }
      }

      let text = `Pos: ${pos.x.toFixed(2)} ${pos.y.toFixed(2)} ${pos.z.toFixed(2)}\nRot: ${rotDeg.x.toFixed(0)} ${rotDeg.y.toFixed(0)} ${rotDeg.z.toFixed(0)}`;
      if (this.rightGripDown) text += `\nZ-Rot: ${this.rightZAxisRotation.toFixed(1)}`;
      if (this.rightHandInfoText) this.rightHandInfoText.setAttribute('value', text);

      rightController.position = { x: pos.x, y: pos.y, z: pos.z };
      rightController.rotation = rotDeg;
      rightController.quaternion = {
        x: this.rightHand.object3D.quaternion.x,
        y: this.rightHand.object3D.quaternion.y,
        z: this.rightHand.object3D.quaternion.z,
        w: this.rightHand.object3D.quaternion.w
      };
      rightController.trigger = this.rightTriggerDown ? 1 : 0;
      rightController.gripActive = this.rightGripDown;

      // Get thumbstick from tracked-controls
      if (this.rightHand.components && this.rightHand.components['tracked-controls']) {
        const gamepad = this.rightHand.components['tracked-controls'].controller?.gamepad;
        if (gamepad) {
          rightController.thumbstick = {
            x: gamepad.axes[2] || 0,
            y: gamepad.axes[3] || 0
          };
          rightController.buttons = {
            a: !!gamepad.buttons[3]?.pressed,
            b: !!gamepad.buttons[4]?.pressed,
            squeeze: !!gamepad.buttons[1]?.pressed,
            thumbstick: !!gamepad.buttons[2]?.pressed
          };
        }
      }
    }

    // Headset
    if (this.headset && this.headset.object3D) {
      const pos = this.headset.object3D.position;
      const rot = this.headset.object3D.rotation;
      headset.position = { x: pos.x, y: pos.y, z: pos.z };
      headset.rotation = {
        x: THREE.MathUtils.radToDeg(rot.x),
        y: THREE.MathUtils.radToDeg(rot.y),
        z: THREE.MathUtils.radToDeg(rot.z)
      };
      headset.quaternion = {
        x: this.headset.object3D.quaternion.x,
        y: this.headset.object3D.quaternion.y,
        z: this.headset.object3D.quaternion.z,
        w: this.headset.object3D.quaternion.w
      };

      if (this.headsetInfoText) {
        const text = `Pos: ${pos.x.toFixed(2)} ${pos.y.toFixed(2)} ${pos.z.toFixed(2)}`;
        this.headsetInfoText.setAttribute('value', text);
      }
    }

    // Send data
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      const hasLeft = leftController.position !== null;
      const hasRight = rightController.position !== null;
      const hasHeadset = headset.position !== null;

      if (hasLeft || hasRight || hasHeadset) {
        const data = {
          timestamp: Date.now(),
          leftController: leftController,
          rightController: rightController,
          headset: headset
        };
        this.websocket.send(JSON.stringify(data));
      }
    }
  }
});

// Add component to scene when loaded
document.addEventListener('DOMContentLoaded', () => {
  const scene = document.querySelector('a-scene');

  if (scene) {
    scene.addEventListener('controllerconnected', (evt) => {
      console.log('Controller CONNECTED:', evt.detail.name, evt.detail.component.data.hand);
    });
    scene.addEventListener('controllerdisconnected', (evt) => {
      console.log('Controller DISCONNECTED:', evt.detail.name, evt.detail.component.data.hand);
    });

    if (scene.hasLoaded) {
      scene.setAttribute('controller-updater', '');
      console.log("controller-updater component added.");
    } else {
      scene.addEventListener('loaded', () => {
        scene.setAttribute('controller-updater', '');
        console.log("controller-updater component added after scene loaded.");
      });
    }
  } else {
    console.error('A-Frame scene not found!');
  }

  addControllerTrackingButton();
});

function addControllerTrackingButton() {
  if (navigator.xr) {
    navigator.xr.isSessionSupported('immersive-ar').then((supported) => {
      if (supported) {
        const startButton = document.createElement('button');
        startButton.id = 'start-tracking-button';
        startButton.textContent = 'Start Controller Tracking';
        startButton.style.position = 'fixed';
        startButton.style.top = '50%';
        startButton.style.left = '50%';
        startButton.style.transform = 'translate(-50%, -50%)';
        startButton.style.padding = '20px 40px';
        startButton.style.fontSize = '20px';
        startButton.style.fontWeight = 'bold';
        startButton.style.backgroundColor = '#4CAF50';
        startButton.style.color = 'white';
        startButton.style.border = 'none';
        startButton.style.borderRadius = '8px';
        startButton.style.cursor = 'pointer';
        startButton.style.zIndex = '9999';
        startButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';

        startButton.onclick = () => {
          const sceneEl = document.querySelector('a-scene');
          if (sceneEl) {
            sceneEl.enterVR(true).catch((err) => {
              console.error('Failed to enter VR:', err);
              alert(`Failed to start AR session: ${err.message}`);
            });
          }
        };

        document.body.appendChild(startButton);

        const sceneEl = document.querySelector('a-scene');
        if (sceneEl) {
          sceneEl.addEventListener('enter-vr', () => {
            startButton.style.display = 'none';
          });
          sceneEl.addEventListener('exit-vr', () => {
            startButton.style.display = 'block';
          });
        }
      }
    }).catch((err) => {
      console.error('Error checking AR support:', err);
    });
  }
}
