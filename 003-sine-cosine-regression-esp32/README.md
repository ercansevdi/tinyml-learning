# 003 — Sine and Cosine Regression on ESP32

This educational TinyML project predicts **sine and cosine from a single angle in radians**. A trained model is converted to INT8 and deployed on an ESP32 Dev Module using PlatformIO and the Arduino framework.

## Validated results

| Measurement | Result |
|---|---:|
| Final INT8 model size | **3,816 bytes** |
| Final model FNV-1a fingerprint | `B64E1E67` |
| Python sine MAE / maximum absolute error | 0.007230 / 0.025874 |
| Python cosine MAE / maximum absolute error | 0.009633 / 0.030964 |
| Python joint tolerance success | **450/450 test samples** |
| ESP32 joint tolerance success | **11/11 tested angles** |
| Maximum ESP32 sine error at tested angles | 0.01592 |
| Maximum ESP32 cosine error at tested angles | 0.02677 |
| First reported Invoke() time | 0.618 ms |
| Subsequent reported Invoke() times | 0.279–0.311 ms |

Joint tolerance success means that **both outputs have absolute error <= 0.05**. It is not classification accuracy and does not imply exact predictions. The original Python test set and training notebook were not included in the provided files; the Python statistics above are user-reported results from the development session. The ESP32 rows are preserved in [docs/esp32_results.csv](docs/esp32_results.csv). Eleven points do not guarantee the error bound over every possible angle.

## Quantization compatibility issue

The original 5,320-byte INT8 model gave large errors with the Arduino runtime. Python and ESP32 had matching model fingerprints and were tested with identical raw INT8 inputs, but their raw outputs differed substantially.

Re-converting the existing trained model with the following experimental TensorFlow converter setting removed the large discrepancy:

```python
converter._experimental_disable_per_channel = True
```

No retraining was performed. The final converted model is 3,816 bytes. Some raw ESP32 outputs differed from the Python interpreter by one INT8 step; all eleven tested angles passed the chosen tolerance. This experiment supports a quantization/runtime compatibility explanation; it is not a general comparison of all TFLM implementations. The setting is experimental and may vary between TensorFlow releases.

## Files

- `firmware/`: PlatformIO Arduino project, source, and embedded model header.
- `models/sinus_cosinus_int8_arduino.tflite`: exact model bytes recovered from the uploaded final header.
- `docs/esp32_results.csv`: the reported eleven-angle hardware run.
- `docs/model_manifest.json`: model size and fingerprints.
- `tools/check_model.py`: standard-library check that the header and model file contain identical bytes.

The firmware source was assembled from the final code used in the development conversation. This package has not been freshly compiled or run on hardware by the packaging assistant. Original training code, Float32 model, and dataset are not included.

## Build and upload

1. Install VS Code and PlatformIO.
2. Open **the `firmware` folder**, which contains `platformio.ini`.
3. Build, then upload to an ESP32 Dev Module.
4. Open Serial Monitor at **115200 baud**.
5. Press **EN/RESET** to see startup information.

The configuration uses the same unpinned `espressif32` platform as the successful user setup, and pins `Chirale_TensorFLowLite` to 2.0.0. The exact platform version from the successful run was not recorded.

## Inference path

The input is a raw angle in radians in the range 0 to 2*pi. Preprocessing is already inside the model and must not be repeated externally.

```text
q_input = clip(round(angle_radians / input_scale) + input_zero_point, -128, 127)
y = (q_output - output_zero_point) * output_scale
```

Output element 0 is sine and element 1 is cosine. Values are not clipped to [-1, 1], so small endpoint overshoots remain visible in the error measurement.

The firmware allocates **16 KiB** for the tensor arena. This allocation is not a measurement of the actual required arena or total ESP32 RAM. The firmware prints the actual arena usage at startup. Inference timing covers `Invoke()` only, excluding angle conversion, quantization, serial printing, and the one-second delay. Reported timing is a small sample, not a sustained benchmark.

This project is a learning exercise, not a faster replacement for standard trigonometric functions.
