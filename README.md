# TinyML Learning

A hands-on learning repository containing a series of TinyML experiments and embedded AI projects.

The goal of this repository is to document my progression from basic machine learning deployment to real-world TinyML applications on microcontrollers such as ESP32 and STM32.

> This is a learning repository. Projects progress from simple introductory examples to sensor-based and embedded AI applications.

## Projects

| # | Project | Hardware | Status |
|---|---|---|---|
| 001 | Sine Regression | ESP32 | Completed |

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

## Next Step

INT8 quantized inference on ESP32.