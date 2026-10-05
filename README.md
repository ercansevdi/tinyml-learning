# TinyML Learning

A hands-on learning repository containing a series of TinyML experiments and embedded AI projects.

The goal of this repository is to document my progression from basic machine learning deployment to real-world TinyML applications on microcontrollers such as ESP32 and STM32.

> This is a learning repository. Projects progress from simple introductory examples to sensor-based and embedded AI applications.

## Projects

| # | Project | Hardware | Status |
|---|---|---|---|
| 001 | [Sine Regression](001-sine-regression-esp32) | ESP32 | Completed |
| 002 | [INT8 Cosine Regression](002-cosine-regression-esp32) | ESP32 / ESP-IDF | Device inference verified at five angles |

## 001 — Sine Regression on ESP32

A small neural network was trained to approximate:

`y = sin(x)`

Workflow:

`Python → TensorFlow/Keras → TensorFlow Lite → PlatformIO → ESP32`

ESP32 result for `x = 1.0`:

- Actual: `0.841471`
- Prediction: `0.845712`
- Absolute Error: `0.004241`
- Inference Time: `119 µs`

## 002 — INT8 Cosine Regression on ESP32

A small fully integer neural network approximates `y = cos(x)` using native ESP-IDF 6.1.0 and TensorFlow Lite Micro. The archived model is 3,344 bytes. ESP32 serial output was recorded at five angles, and an independent host reference reproduced the displayed predictions.

The project includes the firmware, model, original training notebook, board configuration, dependency lock file, serial screenshots, and result CSV. The larger endpoint error at `2π` is documented in the [project README](002-cosine-regression-esp32/README.md).

## Next Step

Measure inference latency, evaluate more input points, and improve endpoint calibration for the INT8 cosine experiment.
