#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cv {

constexpr int CV_8UC1 = 0;
constexpr int CV_8UC3 = 1;
constexpr int CV_8UC4 = 2;
constexpr int CV_32F = 3;

constexpr int COLOR_BGR2GRAY = 10;
constexpr int COLOR_BGRA2GRAY = 11;
constexpr int INTER_AREA = 20;

struct Size {
    int width = 0;
    int height = 0;

    Size() = default;
    Size(int w, int h) : width(w), height(h) {}
};

class Scalar {
public:
    Scalar() = default;
    explicit Scalar(double value)
        : values_{value, 0.0, 0.0, 0.0}
    {
    }

    double operator[](int index) const { return values_[index]; }

private:
    std::array<double, 4> values_{0.0, 0.0, 0.0, 0.0};
};

inline int channelsFromType(int type)
{
    switch (type) {
    case CV_8UC1:
    case CV_32F:
        return 1;
    case CV_8UC3:
        return 3;
    case CV_8UC4:
        return 4;
    default:
        return 1;
    }
}

inline std::size_t elementSizeFromType(int type)
{
    switch (type) {
    case CV_32F:
        return sizeof(float);
    case CV_8UC1:
    case CV_8UC3:
    case CV_8UC4:
    default:
        return sizeof(std::uint8_t);
    }
}

class Mat {
public:
    Mat() = default;

    Mat(const Mat& other)
        : rows_(other.rows_), cols_(other.cols_), type_(other.type_), channels_(other.channels_), data_(other.data_)
    {
        refreshDataPointer();
    }

    Mat(Mat&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), type_(other.type_), channels_(other.channels_), data_(std::move(other.data_))
    {
        refreshDataPointer();
        other.data = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        other.channels_ = 1;
        other.type_ = CV_8UC1;
    }

    Mat& operator=(const Mat& other)
    {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            type_ = other.type_;
            channels_ = other.channels_;
            data_ = other.data_;
            refreshDataPointer();
        }
        return *this;
    }

    Mat& operator=(Mat&& other) noexcept
    {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            type_ = other.type_;
            channels_ = other.channels_;
            data_ = std::move(other.data_);
            refreshDataPointer();
            other.data = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
            other.channels_ = 1;
            other.type_ = CV_8UC1;
        }
        return *this;
    }

    Mat(int rows, int cols, int type)
        : rows_(rows), cols_(cols), type_(type), channels_(channelsFromType(type)), data_(static_cast<std::size_t>(rows) * cols * channels_ * elementSizeFromType(type), 0)
    {
        refreshDataPointer();
    }

    Mat(int rows, int cols, int type, void* data, std::size_t step)
        : rows_(rows), cols_(cols), type_(type), channels_(channelsFromType(type)), data_(static_cast<std::size_t>(rows) * cols * channels_ * elementSizeFromType(type), 0)
    {
        const std::size_t rowBytes = static_cast<std::size_t>(cols) * channels_ * elementSizeFromType(type);
        auto* source = static_cast<std::uint8_t*>(data);
        if (source != nullptr) {
            for (int row = 0; row < rows_; ++row) {
                std::memcpy(data_.data() + static_cast<std::size_t>(row) * rowBytes, source + static_cast<std::size_t>(row) * step, rowBytes);
            }
        }
        refreshDataPointer();
    }

    bool empty() const { return data_.empty() || rows_ <= 0 || cols_ <= 0; }
    bool isContinuous() const { return true; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int channels() const { return channels_; }
    int type() const { return type_; }
    std::size_t elemSize() const { return elementSizeFromType(type_); }
    std::size_t step() const { return static_cast<std::size_t>(cols_) * channels_ * elemSize(); }

    Mat clone() const
    {
        Mat copied(rows_, cols_, type_);
        copied.data_ = data_;
        copied.refreshDataPointer();
        return copied;
    }

    void convertTo(Mat& dst, int rtype, double alpha = 1.0, double beta = 0.0) const
    {
        if (empty()) {
            dst = Mat();
            return;
        }

        const int sourceRows = rows_;
        const int sourceCols = cols_;
        const int sourceType = type_;
        const std::vector<std::uint8_t> sourceData = data_;

        dst = Mat(sourceRows, sourceCols, rtype);

        if (sourceType == CV_8UC1 && rtype == CV_32F) {
            auto* source = reinterpret_cast<const std::uint8_t*>(sourceData.data());
            auto* target = reinterpret_cast<float*>(dst.data_.data());
            const std::size_t count = static_cast<std::size_t>(sourceRows) * sourceCols;
            for (std::size_t index = 0; index < count; ++index) {
                target[index] = static_cast<float>(source[index] * alpha + beta);
            }
            dst.refreshDataPointer();
            return;
        }

        if (sourceType == CV_32F && rtype == CV_32F) {
            std::memcpy(dst.data_.data(), sourceData.data(), sourceData.size());
            auto* target = reinterpret_cast<float*>(dst.data_.data());
            const std::size_t count = static_cast<std::size_t>(sourceRows) * sourceCols;
            for (std::size_t index = 0; index < count; ++index) {
                target[index] = static_cast<float>(target[index] * alpha + beta);
            }
            dst.refreshDataPointer();
            return;
        }

        if (sourceType == CV_8UC1 && rtype == CV_8UC1) {
            std::memcpy(dst.data_.data(), sourceData.data(), sourceData.size());
            dst.refreshDataPointer();
            return;
        }

        if (sourceType == CV_32F && rtype == CV_8UC1) {
            auto* source = reinterpret_cast<const float*>(sourceData.data());
            auto* target = reinterpret_cast<std::uint8_t*>(dst.data_.data());
            const std::size_t count = static_cast<std::size_t>(sourceRows) * sourceCols;
            for (std::size_t index = 0; index < count; ++index) {
                const double value = source[index] * alpha + beta;
                target[index] = static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
            }
            dst.refreshDataPointer();
        }
    }

    std::uint8_t* data = nullptr;
    std::vector<std::uint8_t> data_;

private:
    void refreshDataPointer() { data = data_.empty() ? nullptr : data_.data(); }

    int rows_ = 0;
    int cols_ = 0;
    int type_ = CV_8UC1;
    int channels_ = 1;
};

inline Mat operator-(float value, const Mat& mat)
{
    Mat result(mat.rows(), mat.cols(), mat.type());
    if (mat.empty()) {
        return result;
    }

    if (mat.type() == CV_32F) {
        const auto* source = reinterpret_cast<const float*>(mat.data);
        auto* target = reinterpret_cast<float*>(result.data_.data());
        const std::size_t count = static_cast<std::size_t>(mat.rows()) * mat.cols();
        for (std::size_t index = 0; index < count; ++index) {
            target[index] = value - source[index];
        }
        result.data = result.data_.empty() ? nullptr : result.data_.data();
        return result;
    }

    const auto* source = reinterpret_cast<const std::uint8_t*>(mat.data);
    auto* target = reinterpret_cast<std::uint8_t*>(result.data_.data());
    const std::size_t count = static_cast<std::size_t>(mat.rows()) * mat.cols();
    for (std::size_t index = 0; index < count; ++index) {
        target[index] = static_cast<std::uint8_t>(std::clamp(value - source[index], 0.0f, 255.0f));
    }
    result.data = result.data_.empty() ? nullptr : result.data_.data();
    return result;
}

inline Scalar mean(const Mat& mat)
{
    if (mat.empty()) {
        return Scalar(0.0);
    }

    const std::size_t count = static_cast<std::size_t>(mat.rows()) * mat.cols() * mat.channels();
    double sum = 0.0;

    if (mat.type() == CV_32F) {
        const auto* source = reinterpret_cast<const float*>(mat.data);
        for (std::size_t index = 0; index < count; ++index) {
            sum += source[index];
        }
    } else {
        const auto* source = reinterpret_cast<const std::uint8_t*>(mat.data);
        for (std::size_t index = 0; index < count; ++index) {
            sum += source[index];
        }
    }

    return Scalar(sum / static_cast<double>(count));
}

} // namespace cv

constexpr int CV_8UC1 = cv::CV_8UC1;
constexpr int CV_8UC3 = cv::CV_8UC3;
constexpr int CV_8UC4 = cv::CV_8UC4;
constexpr int CV_32F = cv::CV_32F;
constexpr int COLOR_BGR2GRAY = cv::COLOR_BGR2GRAY;
constexpr int COLOR_BGRA2GRAY = cv::COLOR_BGRA2GRAY;
constexpr int INTER_AREA = cv::INTER_AREA;