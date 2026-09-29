#include <Arduino.h>
#include <ArduTFLite.h>
#include "sine_model_float_data.h"
#include <math.h>

constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println("TinyML Sinus ESP32");

    if (!modelInit(
        sine_model_float_data,
        tensor_arena,
        kTensorArenaSize
    )) {
        Serial.println("Model baslatilamadi!");
        while (true);
    }

    Serial.println("Model hazir.");
}

void loop() {
    float x = 1.0f;

    modelSetInput(x, 0);

    unsigned long baslangic = micros();

    if (!modelRunInference()) {
        Serial.println("Inference hatasi!");
        delay(3000);
        return;
    }

    unsigned long sure = micros() - baslangic;

    float tahmin = modelGetOutput(0);
    float gercek = sin(x);
    float hata = fabs(gercek - tahmin);

    Serial.println("------------------");

    Serial.print("x      : ");
    Serial.println(x, 6);

    Serial.print("Gercek : ");
    Serial.println(gercek, 6);

    Serial.print("Tahmin : ");
    Serial.println(tahmin, 6);

    Serial.print("Hata   : ");
    Serial.println(hata, 6);

    Serial.print("Sure   : ");
    Serial.print(sure);
    Serial.println(" us");

    delay(3000);
}