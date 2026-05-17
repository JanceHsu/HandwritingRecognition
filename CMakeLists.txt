cmake_minimum_required(VERSION 3.14)
project(DigitRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(DIGIT_RECOG_DEFAULT_DEVICE "auto" CACHE STRING "Default inference device: cpu|cuda|auto")
set_property(CACHE DIGIT_RECOG_DEFAULT_DEVICE PROPERTY STRINGS cpu cuda auto)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Torch REQUIRED)

add_executable(digit_recog
    src/main.cpp
    src/mainwindow.cpp
    src/canvas.cpp
    src/recognizer.cpp
)

target_include_directories(digit_recog PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(digit_recog PRIVATE
    Qt6::Widgets
    ${TORCH_LIBRARIES}
)

target_compile_definitions(digit_recog PRIVATE
    DIGIT_RECOG_DEFAULT_DEVICE="${DIGIT_RECOG_DEFAULT_DEVICE}"
)

target_compile_features(digit_recog PRIVATE cxx_std_17)