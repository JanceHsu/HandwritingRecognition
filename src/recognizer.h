#pragma once

#ifdef slots
#pragma push_macro("slots")
#undef slots
#define HANDWRITING_RECOGNITION_RESTORE_SLOTS_MACRO
#endif

#ifdef signals
#pragma push_macro("signals")
#undef signals
#define HANDWRITING_RECOGNITION_RESTORE_SIGNALS_MACRO
#endif

#include <torch/script.h>

#ifdef HANDWRITING_RECOGNITION_RESTORE_SIGNALS_MACRO
#pragma pop_macro("signals")
#undef HANDWRITING_RECOGNITION_RESTORE_SIGNALS_MACRO
#endif

#ifdef HANDWRITING_RECOGNITION_RESTORE_SLOTS_MACRO
#pragma pop_macro("slots")
#undef HANDWRITING_RECOGNITION_RESTORE_SLOTS_MACRO
#endif

#include <QImage>

#include <string>
#include <vector>

struct PredictResult {
    int digit = -1;
    float confidence = 0.0f;
};

class DigitRecognizer {
public:
    explicit DigitRecognizer(const std::string& modelPath, bool useCuda);
    int predict(const QImage& inputImage);
    PredictResult predictWithConfidence(const QImage& inputImage);
    const std::string& deviceName() const;
    void warmUp();

private:
    torch::jit::script::Module model;
    torch::Device device = torch::kCPU;
    std::string deviceName_ = "cpu";
};
