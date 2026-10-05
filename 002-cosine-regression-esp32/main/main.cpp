#include <stdio.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// Modelin ara hesaplamalari icin 16 KiB RAM.
alignas(16) static uint8_t tensor_arena[16 * 1024];

static void cosinus_task(void *)
{
    tflite::InitializeTarget();

    // Basliktaki model verisini kullan.
    const tflite::Model *model = tflite::GetModel(model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("HATA: Model semasi uyumsuz.\n");
        vTaskDelete(nullptr);
        return;
    }

    // Modelin Dense katmanlari bu islemi kullanir.
    static tflite::MicroMutableOpResolver<1> resolver;

    if (resolver.AddFullyConnected() != kTfLiteOk) {
        printf("HATA: FullyConnected eklenemedi.\n");
        vTaskDelete(nullptr);
        return;
    }

    // Modeli calistiracak yorumlayici.
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("HATA: Tensor bellegi ayrilamadi.\n");
        vTaskDelete(nullptr);
        return;
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    if (!input || !output ||
        input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        input->bytes != 1 || output->bytes != 1) {
        printf("HATA: Tek giris/cikisli INT8 model bekleniyor.\n");
        vTaskDelete(nullptr);
        return;
    }

    // Acilar radyan: 0, pi/2, pi, 3*pi/2, 2*pi.
    const float test_x[] = {
        0.0f, 1.5707963f, 3.1415927f, 4.7123890f, 6.2831853f
    };
    unsigned index = 0;

    while (1) {
        float x = test_x[index];

        // Gercek sayiyi modelin INT8 girisine donustur.
        long q = lroundf(x / input->params.scale)
                 + input->params.zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input->data.int8[0] = static_cast<int8_t>(q);

        // Sinir agini calistir.
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("HATA: Model tahmini basarisiz.\n");
            vTaskDelete(nullptr);
            return;
        }

        // INT8 cikisini tekrar gercek sayiya donustur.
        int q_y = static_cast<int>(output->data.int8[0]);
        float prediction = (q_y - output->params.zero_point)
                           * output->params.scale;

        // Gercek kosinus, tahmini kontrol etmek icin.
        float actual = cosf(x);
        float error = fabsf(prediction - actual);

        printf("x=%.4f rad | q=%d | Model=%.5f | Gercek=%.5f | Hata=%.5f\n",
               x, static_cast<int>(q), prediction, actual, error);

        index = (index + 1) % 5;
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

extern "C" void app_main(void)
{
    // Model gorevi icin 8 KiB stack.
    if (xTaskCreate(cosinus_task, "cosinus", 8 * 1024,
                    nullptr, 5, nullptr) != pdPASS) {
        printf("HATA: Model gorevi baslatilamadi.\n");
    }
}
