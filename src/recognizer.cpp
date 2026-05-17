#include "recognizer.h"

#include <QImage>

#include <opencv2/imgproc.hpp>
#include <torch/cuda.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> preprocessImage(const QImage& img)
{
    if (img.isNull()) {
        return {};
    }

    const QImage gray = img.convertToFormat(QImage::Format_Grayscale8).scaled(
        28,
        28,
        Qt::IgnoreAspectRatio,
        Qt::SmoothTransformation
    );

    if (gray.isNull() || gray.width() != 28 || gray.height() != 28) {
        return {};
    }

    std::vector<float> normalized(28 * 28, 0.0f);
    for (int row = 0; row < gray.height(); ++row) {
        const auto* line = gray.constScanLine(row);
        for (int col = 0; col < gray.width(); ++col) {
            normalized[static_cast<std::size_t>(row) * 28 + static_cast<std::size_t>(col)] =
                1.0f - static_cast<float>(line[col]) / 255.0f;
        }
    }

    return normalized;
}

std::string normalizeDeviceName(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string requestedDeviceName()
{
    const char* envDevice = std::getenv("LIBTORCH_DEVICE");
    if (envDevice != nullptr && envDevice[0] != '\0') {
        return normalizeDeviceName(std::string(envDevice));
    }

#ifdef DIGIT_RECOG_DEFAULT_DEVICE
    return normalizeDeviceName(std::string(DIGIT_RECOG_DEFAULT_DEVICE));
#else
    return "cpu";
#endif
}

} // namespace

torch::Device DigitRecognizer::resolveDevice()
{
    const std::string requested = requestedDeviceName();

    if (requested == "cuda") {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("LIBTORCH_DEVICE=cuda but CUDA is not available in current LibTorch/runtime.");
        }
        deviceName_ = "cuda";
        return torch::Device(torch::kCUDA);
    }

    if (requested == "auto") {
        if (torch::cuda::is_available()) {
            deviceName_ = "cuda";
            return torch::Device(torch::kCUDA);
        }
        deviceName_ = "cpu";
        return torch::Device(torch::kCPU);
    }

    if (requested != "cpu") {
        throw std::runtime_error("Invalid LIBTORCH_DEVICE value. Use cpu, cuda, or auto.");
    }

    deviceName_ = "cpu";
    return torch::Device(torch::kCPU);
}

DigitRecognizer::DigitRecognizer(const std::string& modelPath)
{
    try {
        device = resolveDevice();
        model = torch::jit::load(modelPath, device);
        model.eval();
    } catch (const c10::Error& error) {
        throw std::runtime_error(std::string("Failed to load TorchScript model: ") + error.what());
    }
}

const std::string& DigitRecognizer::deviceName() const
{
    return deviceName_;
}

void DigitRecognizer::warmUp()
{
    if (device.type() != torch::kCUDA) {
        return;
    }

    torch::InferenceMode guard;
    auto warmInput = torch::zeros(
        {1, 1, 28, 28},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(warmInput);
    (void)model.forward(inputs).toTensor();
    torch::cuda::synchronize();
}

std::vector<float> DigitRecognizer::preprocess(const cv::Mat& img)
{
    if (img.empty()) {
        return {};
    }

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = img.clone();
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(28, 28), 0.0, 0.0, cv::INTER_AREA);

    if (resized.empty() || resized.type() != CV_8UC1) {
        return {};
    }

    std::vector<float> normalized(28 * 28, 0.0f);
    const auto* source = reinterpret_cast<const std::uint8_t*>(resized.data);
    for (std::size_t index = 0; index < normalized.size(); ++index) {
        normalized[index] = 1.0f - static_cast<float>(source[index]) / 255.0f;
    }

    return normalized;
}

int DigitRecognizer::predict(const cv::Mat& inputImage)
{
    std::vector<float> processed = preprocess(inputImage);
    if (processed.size() != 28 * 28) {
        throw std::runtime_error("Input image is empty");
    }

    auto input = torch::from_blob(
        processed.data(),
        {1, 1, 28, 28},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    input = input.to(device);

    torch::InferenceMode guard;
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input);

    torch::Tensor logits = model.forward(inputs).toTensor().to(torch::kCPU);
    return static_cast<int>(logits.argmax(1).item<int64_t>());
}

int DigitRecognizer::predict(const QImage& inputImage)
{
    std::vector<float> processed = preprocessImage(inputImage);
    if (processed.size() != 28 * 28) {
        throw std::runtime_error("Input image is empty");
    }

    auto input = torch::from_blob(
        processed.data(),
        {1, 1, 28, 28},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    input = input.to(device);

    torch::InferenceMode guard;
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input);

    torch::Tensor logits = model.forward(inputs).toTensor().to(torch::kCPU);
    return static_cast<int>(logits.argmax(1).item<int64_t>());
}