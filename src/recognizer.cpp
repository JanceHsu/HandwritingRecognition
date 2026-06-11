#include "recognizer.h"

#include <QImage>
#include <QPainter>

#include <torch/cuda.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kMnistMean = 0.1307f;
constexpr float kMnistStd = 0.3081f;

std::vector<float> normalizeToMnist(const QImage& grayImage)
{
    QRect inkBounds;
    for (int row = 0; row < grayImage.height(); ++row) {
        const auto* line = grayImage.constScanLine(row);
        for (int col = 0; col < grayImage.width(); ++col) {
            if (static_cast<unsigned char>(line[col]) < 245) {
                if (inkBounds.isNull()) {
                    inkBounds = QRect(col, row, 1, 1);
                } else {
                    inkBounds.setLeft(std::min(inkBounds.left(), col));
                    inkBounds.setTop(std::min(inkBounds.top(), row));
                    inkBounds.setRight(std::max(inkBounds.right(), col));
                    inkBounds.setBottom(std::max(inkBounds.bottom(), row));
                }
            }
        }
    }

    inkBounds = inkBounds.adjusted(-4, -4, 4, 4).intersected(grayImage.rect());
    const int side = std::max(inkBounds.width(), inkBounds.height());

    QImage padded(side, side, QImage::Format_Grayscale8);
    padded.fill(Qt::white);

    QPainter painter(&padded);
    const int offsetX = (side - inkBounds.width()) / 2;
    const int offsetY = (side - inkBounds.height()) / 2;
    painter.drawImage(QPoint(offsetX, offsetY), grayImage.copy(inkBounds));
    painter.end();

    const QImage resized = padded.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    std::vector<float> normalized(28 * 28, 0.0f);
    for (int row = 0; row < resized.height(); ++row) {
        const auto* line = resized.constScanLine(row);
        for (int col = 0; col < resized.width(); ++col) {
            const float pixel = 1.0f - static_cast<float>(line[col]) / 255.0f;
            normalized[static_cast<std::size_t>(row) * 28 + static_cast<std::size_t>(col)] =
                (pixel - kMnistMean) / kMnistStd;
        }
    }

    return normalized;
}

} // namespace

DigitRecognizer::DigitRecognizer(const std::string& modelPath, bool useCuda)
{
    if (!std::filesystem::exists(modelPath)) {
        throw std::runtime_error("Model file not found: " + modelPath);
    }

    if (useCuda) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA requested but not available in current LibTorch/runtime.");
        }
        device = torch::Device(torch::kCUDA);
        deviceName_ = "cuda";
    } else {
        device = torch::Device(torch::kCPU);
        deviceName_ = "cpu";
    }

    model = torch::jit::load(modelPath, device);
    model.eval();
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

int DigitRecognizer::predict(const QImage& inputImage)
{
    const QImage gray = inputImage.convertToFormat(QImage::Format_Grayscale8);
    std::vector<float> processed = normalizeToMnist(gray);
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

PredictResult DigitRecognizer::predictWithConfidence(const QImage& inputImage)
{
    const QImage gray = inputImage.convertToFormat(QImage::Format_Grayscale8);
    std::vector<float> processed = normalizeToMnist(gray);
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
    torch::Tensor probs = torch::softmax(logits, 1);

    PredictResult result;
    result.digit = static_cast<int>(probs.argmax(1).item<int64_t>());
    result.confidence = probs[0][result.digit].item<float>();
    return result;
}
