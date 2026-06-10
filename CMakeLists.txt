cmake_minimum_required(VERSION 3.14)
project(HandwritingRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia)
find_package(Torch REQUIRED)

add_executable(handwriting_recog
    src/main.cpp
    src/airwritecontroller.cpp
    src/mainwindow.cpp
    src/canvas.cpp
    src/recognizer.cpp
)

target_include_directories(handwriting_recog PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(handwriting_recog PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    ${TORCH_LIBRARIES}
)

add_custom_command(TARGET handwriting_recog POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/artifacts/models
            $<TARGET_FILE_DIR:handwriting_recog>/models
    COMMENT "Copying MNIST model to target output directory"
)

if (WIN32)
    target_link_libraries(handwriting_recog PRIVATE
        mf
        mfplat
        mfreadwrite
        mfuuid
        ole32
        shlwapi
    )
endif()

target_compile_features(handwriting_recog PRIVATE cxx_std_17)