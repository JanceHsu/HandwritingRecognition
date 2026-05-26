#include "recognizer.h"

#include <QImage>
#include <QPainter>

#include <opencv2/imgproc.hpp>
#include <torch/cuda.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kMnistMean = 0.1307f;
constexpr float kMnistStd = 0.3081f;

std::vector<float> normalizeToMnist(const QImage& grayImage)
{
    if (grayImage.isNull() || grayImage.width() <= 0 || grayImage.height() <= 0) {
        return {};
    }

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

    if (inkBounds.isNull()) {
        return {};
    }

    inkBounds = inkBounds.adjusted(-4, -4, 4, 4).intersected(grayImage.rect());
    const int side = std::max(inkBounds.width(), inkBounds.height());
    if (side <= 0) {
        return {};
    }

    QImage padded(side, side, QImage::Format_Grayscale8);
    padded.fill(Qt::white);

    QPainter painter(&padded);
    const int offsetX = (side - inkBounds.width()) / 2;
    const int offsetY = (side - inkBounds.height()) / 2;
    painter.drawImage(QPoint(offsetX, offsetY), grayImage.copy(inkBounds));
    painter.end();

    const QImage resized = padded.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    if (resized.isNull() || resized.width() != 28 || resized.height() != 28) {
        return {};
    }

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

std::vector<float> preprocessImage(const QImage& img)
{
    if (img.isNull()) {
        return {};
    }

    return normalizeToMnist(img.convertToFormat(QImage::Format_Grayscale8));
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

#ifdef HANDWRITING_RECOG_DEFAULT_DEVICE
    return normalizeDeviceName(std::string(HANDWRITING_RECOG_DEFAULT_DEVICE));
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

    if (gray.empty() || gray.type() != CV_8UC1) {
        return {};
    }

    const int rows = gray.rows();
    const int cols = gray.cols();
    const auto* source = reinterpret_cast<const std::uint8_t*>(gray.data);

    int left = cols;
    int top = rows;
    int right = -1;
    int bottom = -1;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const std::uint8_t pixel = source[static_cast<std::size_t>(row) * cols + static_cast<std::size_t>(col)];
            if (pixel < 245) {
                left = std::min(left, col);
                top = std::min(top, row);
                right = std::max(right, col);
                bottom = std::max(bottom, row);
            }
        }
    }

    if (right < left || bottom < top) {
        return {};
    }

    left = std::max(0, left - 4);
    top = std::max(0, top - 4);
    right = std::min(cols - 1, right + 4);
    bottom = std::min(rows - 1, bottom + 4);

    const int inkWidth = right - left + 1;
    const int inkHeight = bottom - top + 1;
    const int side = std::max(inkWidth, inkHeight);
    if (side <= 0) {
        return {};
    }

    cv::Mat padded(side, side, CV_8UC1);
    std::fill(padded.data, padded.data + static_cast<std::size_t>(side) * side, 255);

    const int offsetX = (side - inkWidth) / 2;
    const int offsetY = (side - inkHeight) / 2;
    auto* paddedData = reinterpret_cast<std::uint8_t*>(padded.data);
    for (int row = 0; row < inkHeight; ++row) {
        const auto* sourceLine = source + static_cast<std::size_t>(top + row) * cols + left;
        auto* targetLine = paddedData + static_cast<std::size_t>(offsetY + row) * side + offsetX;
        std::memcpy(targetLine, sourceLine, static_cast<std::size_t>(inkWidth));
    }

    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(28, 28), 0.0, 0.0, cv::INTER_AREA);
    if (resized.empty() || resized.type() != CV_8UC1) {
        return {};
    }

    std::vector<float> normalized(28 * 28, 0.0f);
    const int resizedRows = resized.rows();
    const int resizedCols = resized.cols();
    const auto* resizedData = reinterpret_cast<const std::uint8_t*>(resized.data);
    for (int row = 0; row < resizedRows; ++row) {
        for (int col = 0; col < resizedCols; ++col) {
            const std::uint8_t pixel = resizedData[static_cast<std::size_t>(row) * resizedCols + static_cast<std::size_t>(col)];
            const float ink = 1.0f - static_cast<float>(pixel) / 255.0f;
            normalized[static_cast<std::size_t>(row) * 28 + static_cast<std::size_t>(col)] =
                (ink - kMnistMean) / kMnistStd;
        }
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