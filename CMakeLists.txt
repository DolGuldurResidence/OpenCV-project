cmake_minimum_required(VERSION 3.14) 
project(OpenCVProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Включаем модуль FetchContent
include(FetchContent)

# Указываем репозиторий OpenCV (можно выбрать версию)
FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.9.0  # Можно указать нужную версию (4.5.5, 4.8.0 и т. д.)
)


FetchContent_MakeAvailable(opencv)

Если нужен opencv_contrib (доп. модули)
FetchContent_Declare(
     opencv_contrib
     GIT_REPOSITORY https://github.com/opencv/opencv_contrib.git
     GIT_TAG 4.x
)
FetchContent_MakeAvailable(opencv_contrib)

# Создаём исполняемый файл
add_executable(${PROJECT_NAME} main.cpp)



Можно линковать все модули одной строкой (если нужно)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})