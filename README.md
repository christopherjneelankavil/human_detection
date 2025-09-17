## Human Detection Robot (Raspberry Pi)

A Raspberry Pi–powered mobile robot that can:

- Detect and track people using a TensorFlow Lite object detection model
- Navigate autonomously with ultrasonic obstacle avoidance as a fallback
- Drive differential motors via GPIO with speed control (PWM)
- Visualize traversed path as a heatmap (for the wall-following demo)

### Repository layout

- `test1.py` — Person detection and tracking with TFLite; commands motors to follow a person while maintaining distance
- `sensor_automated_control.py_decrypted.py` — Person detection + obstacle avoidance + timeout-driven navigation without a person
- `automated_control.py_decrypted.py` — Alternative integrated control (variant)
- `test.py` — Ultrasonic-only wall-following style navigation with path heatmap
- `fallback.py` — Minimal obstacle-avoidance navigation loop
- `decrypt.py` — Decrypt `.enc` files into `*_decrypted.py` using AES-CBC
- `models/` — TFLite model and labels
  - `mobilenet_ssd_v2_coco_quant_postprocess.tflite`
  - `coco_labels.txt`

Encrypted files provided (optional):

- `face.py.enc`, `installation_wizard.py.enc`, `person_detection.py.enc`, `test.py.enc`

### Hardware requirements

- Raspberry Pi with GPIO (tested conceptually with `gpiozero`)
- Dual H-bridge motor driver and two DC motors
- Ultrasonic distance sensor (HC-SR04 or compatible)
- USB camera (for TFLite detection)

### GPIO pin mapping (as used across scripts)

- Left motor: `IN3=12`, `IN4=16`, `IN1=20`, `IN2=21`, `ENB=18 (PWM)`, `ENB2=13 (PWM)`
- Right motor: `IN1_2=6`, `IN2_2=5`, `IN3_2=22`, `IN4_2=27`, `ENA_2=23 (PWM)`, `ENB_2=24 (PWM)`
- Ultrasonic sensor: `TRIGGER=17`, `ECHO=27`

Adjust pins to match your wiring if needed.

### Software requirements

- Python 3.9+
- Packages:
  - `numpy`, `matplotlib`
  - `opencv-python`, `Pillow`
  - `gpiozero`
  - TensorFlow Lite runtime (one of):
    - `tflite-runtime` or `tensorflow` (includes TFLite)
  - For decryption utility: `pycryptodome`

Example install on Raspberry Pi OS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib opencv-python Pillow gpiozero tflite-runtime pycryptodome
```

Note: on non-RPi hosts, `gpiozero` will fail without GPIO. Use a Raspberry Pi for motor/sensor control.

### Models

The `models/` folder contains the COCO MobileNet SSD v2 quantized TFLite model and labels. If you replace the model, keep input/output tensor shapes compatible or update the code accordingly.

### Running the demos

Prerequisites:

- Wire the motors and ultrasonic sensor according to the pin map above
- Connect a camera (`/dev/video0` assumed)
- Ensure the `models/` folder is present

1. Person detection and tracking (with distance safety)

Runs a TFLite detector, counts persons, and follows while keeping safe distance based on bounding-box area ratio.

```bash
python test1.py
```

Controls used:

- Forward when person is centered and sufficiently far
- Left/right turns based on horizontal deviation (`x_deviation`)
- Stops when area ratio exceeds `area_threshold` (too close)

2. Person detection with obstacle-aware automation and timeout navigation

Starts with person detection; if no person appears for 10 seconds, switches to navigation using ultrasonic obstacle avoidance until a person is seen again.

```bash
python sensor_automated_control.py_decrypted.py
```

Key parameters:

- `PERSON_DETECTION_TIMEOUT = 10` (seconds)
- `OBSTACLE_DISTANCE_THRESHOLD = 0.3` (meters)

3. Ultrasonic-only wall-following navigation with path heatmap

Continuously moves forward, turns left/right to avoid obstacles, and logs visited cells into a `1000x1000` grid. On `Ctrl+C`, shows a heatmap of the path.

```bash
python test.py
```

Notes:

- Heading is maintained in degrees (0, 90, 180, 270) and translated via `headingMap`
- Grid indices are stored as `movementArray[y][x]`

4. Minimal obstacle-avoidance fallback loop

Simple loop to demonstrate ultrasonic-based avoidance without camera logic.

```bash
python fallback.py
```

5. Decrypting encrypted scripts

The repository includes some `.enc` files encrypted with AES-CBC. Use `decrypt.py` to produce `*_decrypted.py` files.

```bash
python decrypt.py
# Enter the path to an .enc file when prompted
# Provide the decryption key when asked
```

The script pads/trims the key to 32 bytes and writes `<name>_decrypted.py` alongside the source file.

### Tuning and customization

- Motor speed: `speed`/`SPEED` variables (default ~0.7)
- Detection threshold: `threshold = 0.3` in vision scripts
- Max detections: `top_k = 5`
- Area-stop threshold: `area_threshold = 0.85` in person-tracking logic
- Ultrasonic thresholds: `OBSTACLE_DISTANCE_THRESHOLD` (in cm for some scripts, meters in others—keep consistent if you refactor)

### Troubleshooting

- No camera frames: ensure `/dev/video0` exists; try `cv2.VideoCapture(0)` on the Pi desktop first
- GPIO permission errors: run with proper permissions or configure `gpiozero`/`pigpio` correctly
- Model load errors: verify files in `models/` and correct paths
- Motors not moving as expected: double-check pin mapping and H-bridge wiring; some `INx` pins may need inversion depending on driver

### License

This project is educational. Review model and dependency licenses for your use case.
