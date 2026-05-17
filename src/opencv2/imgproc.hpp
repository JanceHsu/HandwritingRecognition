#pragma once

#include "core.hpp"

#include <algorithm>
#include <cmath>

namespace cv {

inline void cvtColor(const Mat& src, Mat& dst, int code)
{
    if (src.empty()) {
        dst = Mat();
        return;
    }

    if (code == COLOR_BGR2GRAY && src.channels() == 3) {
        dst = Mat(src.rows(), src.cols(), CV_8UC1);
        const auto* source = reinterpret_cast<const std::uint8_t*>(src.data);
        auto* target = reinterpret_cast<std::uint8_t*>(dst.data_.data());
        for (int row = 0; row < src.rows(); ++row) {
            for (int col = 0; col < src.cols(); ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * src.cols() * 3 + static_cast<std::size_t>(col) * 3;
                const double b = source[index + 0];
                const double g = source[index + 1];
                const double r = source[index + 2];
                target[static_cast<std::size_t>(row) * src.cols() + col] = static_cast<std::uint8_t>(std::clamp(0.114 * b + 0.587 * g + 0.299 * r, 0.0, 255.0));
            }
        }
        dst.data = dst.data_.empty() ? nullptr : dst.data_.data();
        return;
    }

    if (code == COLOR_BGRA2GRAY && src.channels() == 4) {
        dst = Mat(src.rows(), src.cols(), CV_8UC1);
        const auto* source = reinterpret_cast<const std::uint8_t*>(src.data);
        auto* target = reinterpret_cast<std::uint8_t*>(dst.data_.data());
        for (int row = 0; row < src.rows(); ++row) {
            for (int col = 0; col < src.cols(); ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * src.cols() * 4 + static_cast<std::size_t>(col) * 4;
                const double b = source[index + 0];
                const double g = source[index + 1];
                const double r = source[index + 2];
                target[static_cast<std::size_t>(row) * src.cols() + col] = static_cast<std::uint8_t>(std::clamp(0.114 * b + 0.587 * g + 0.299 * r, 0.0, 255.0));
            }
        }
        dst.data = dst.data_.empty() ? nullptr : dst.data_.data();
        return;
    }

    dst = src.clone();
}

inline void resize(const Mat& src, Mat& dst, const Size& size, double = 0.0, double = 0.0, int = INTER_AREA)
{
    if (src.empty()) {
        dst = Mat();
        return;
    }

    dst = Mat(size.height, size.width, src.type());
    const int channels = src.channels();
    const double scaleY = static_cast<double>(src.rows()) / size.height;
    const double scaleX = static_cast<double>(src.cols()) / size.width;

    if (src.type() == CV_8UC1) {
        const auto* source = reinterpret_cast<const std::uint8_t*>(src.data);
        auto* target = reinterpret_cast<std::uint8_t*>(dst.data_.data());
        for (int y = 0; y < size.height; ++y) {
            const int srcY = std::clamp(static_cast<int>(std::floor((y + 0.5) * scaleY)), 0, src.rows() - 1);
            for (int x = 0; x < size.width; ++x) {
                const int srcX = std::clamp(static_cast<int>(std::floor((x + 0.5) * scaleX)), 0, src.cols() - 1);
                target[static_cast<std::size_t>(y) * size.width + x] = source[static_cast<std::size_t>(srcY) * src.cols() + srcX];
            }
        }
    } else {
        const auto* source = reinterpret_cast<const std::uint8_t*>(src.data);
        auto* target = reinterpret_cast<std::uint8_t*>(dst.data_.data());
        const std::size_t rowStride = static_cast<std::size_t>(src.cols()) * channels;
        const std::size_t outStride = static_cast<std::size_t>(size.width) * channels;
        for (int y = 0; y < size.height; ++y) {
            const int srcY = std::clamp(static_cast<int>(std::floor((y + 0.5) * scaleY)), 0, src.rows() - 1);
            for (int x = 0; x < size.width; ++x) {
                const int srcX = std::clamp(static_cast<int>(std::floor((x + 0.5) * scaleX)), 0, src.cols() - 1);
                const std::size_t sourceIndex = static_cast<std::size_t>(srcY) * rowStride + static_cast<std::size_t>(srcX) * channels;
                const std::size_t targetIndex = static_cast<std::size_t>(y) * outStride + static_cast<std::size_t>(x) * channels;
                for (int channel = 0; channel < channels; ++channel) {
                    target[targetIndex + channel] = source[sourceIndex + channel];
                }
            }
        }
    }

    dst.data = dst.data_.empty() ? nullptr : dst.data_.data();
}

} // namespace cv