#include <Arduino.h>
#include <math.h>
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// Modelin ara hesaplamaları için RAM.
alignas(16) static uint8_t tensor_arena[16 * 1024];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;
static bool hazir = false;

// Önce bu açılarda kart sonuçlarını kontrol edeceğiz.
static const float acilar[] = {
    0, 30, 45, 60, 90, 120, 180, 225, 270, 315, 360
};

static const unsigned aci_sayisi =
    sizeof(acilar) / sizeof(acilar[0]);

static unsigned index_aci = 0;
static unsigned basarili = 0;

void setup()
{
    Serial.begin(115200);
    delay(1500);

    Serial.println("\nPROJE 03: SINUS + KOSINUS");

    const tflite::Model* model = tflite::GetModel(g_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("HATA: Model semasi uyumsuz.");
        return;
    }

    // İlk çalıştırmada eksik operatör sorununu önlemek için.
    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter yorumlayici(
        model, resolver, tensor_arena, sizeof(tensor_arena)
    );

    interpreter = &yorumlayici;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("HATA: Tensor ayirma basarisiz.");
        Serial.println("Bu mesajin ustundeki hata satirlarini kontrol et.");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (!input || !output ||
        input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8 ||
        input->bytes != 1 ||
        output->bytes != 2 ||
        input->params.scale <= 0 ||
        output->params.scale <= 0) {
        Serial.println("HATA: Bir giris, iki cikisli INT8 model bekleniyor.");
        return;
    }

    Serial.printf("Model: %u bayt\n",
                  static_cast<unsigned>(sizeof(g_model_data)));

    Serial.printf("Kullanilan arena: %u bayt\n",
                  static_cast<unsigned>(interpreter->arena_used_bytes()));

    Serial.printf("Giris scale=%.9f zero=%d\n",
                  input->params.scale, input->params.zero_point);

    Serial.printf("Cikis scale=%.9f zero=%d\n\n",
                  output->params.scale, output->params.zero_point);

    hazir = true;
}

void loop()
{
    if (!hazir) {
        delay(1000);
        return;
    }

    float derece = acilar[index_aci];
    float radyan = derece * (6.28318530718f / 360.0f);

    // Radyan -> INT8
    long q = lroundf(radyan / input->params.scale)
             + input->params.zero_point;

    if (q < -128) q = -128;
    if (q > 127) q = 127;

    input->data.int8[0] = static_cast<int8_t>(q);

    uint32_t baslangic = micros();
    TfLiteStatus durum = interpreter->Invoke();
    uint32_t sure_us = micros() - baslangic;

    if (durum != kTfLiteOk) {
        Serial.println("HATA: Model calistirilamadi.");
        hazir = false;
        return;
    }

    // INT8 -> gerçek sayı; çıkış sırası [sinüs, kosinüs].
    float sinus =
        (static_cast<int>(output->data.int8[0])
         - output->params.zero_point) * output->params.scale;

    float kosinus =
        (static_cast<int>(output->data.int8[1])
         - output->params.zero_point) * output->params.scale;

    float gercek_sin = sinf(radyan);
    float gercek_cos = cosf(radyan);

    float hata_sin = fabsf(sinus - gercek_sin);
    float hata_cos = fabsf(kosinus - gercek_cos);

    bool uygun = hata_sin <= 0.05f && hata_cos <= 0.05f;
    if (uygun) basarili++;

    Serial.printf(
        "%.0f derece | SIN=%.5f gercek=%.5f hata=%.5f"
        " | COS=%.5f gercek=%.5f hata=%.5f"
        " | %.3f ms | %s\n",
        derece,
        sinus, gercek_sin, hata_sin,
        kosinus, gercek_cos, hata_cos,
        sure_us / 1000.0,
        uygun ? "GECTI" : "TOLERANS DISI"
    );

    index_aci++;

    if (index_aci == aci_sayisi) {
        Serial.printf(
            "\nTUR SONU: %u/%u aci tolerans icinde.\n\n",
            basarili, aci_sayisi
        );

        index_aci = 0;
        basarili = 0;
    }

    delay(1000);
}
