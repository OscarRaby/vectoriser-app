cmake_minimum_required(VERSION 3.15)
project(OrganicVectoriserCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (vcpkg installs as opencv4)
find_package(OpenCV CONFIG REQUIRED)

# Optional: FastNoise2 support for Perlin noise
option(USE_FASTNOISE "Enable FastNoise2 for Perlin noise generation" OFF)

# Source files
set(SOURCES
    SegmentationService.cpp
    ContourService.cpp
    GeometryService.cpp
    BridgingService.cpp
    SortingService.cpp
    SVGWriter.cpp
    ParameterScaler.cpp
    VectorizationPipeline.cpp
    NativeBridge.cpp
)

set(HEADERS
    VectorizationParameters.h
    SegmentationService.h
    ContourService.h
    GeometryService.h
    BridgingService.h
    SortingService.h
    SVGWriter.h
    ParameterScaler.h
    VectorizationPipeline.h
    NativeBridge.h
)

# Create library
add_library(OrganicVectoriser STATIC ${SOURCES} ${HEADERS})

# Native bridge DLL for C# interop
add_library(OrganicVectoriserNative SHARED NativeBridge.cpp)
target_link_libraries(OrganicVectoriserNative PUBLIC OrganicVectoriser ${OpenCV_LIBS})
target_include_directories(OrganicVectoriserNative PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(OrganicVectoriser PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(OrganicVectoriser PUBLIC
    ${OpenCV_LIBS}
)

# Optional FastNoise2
if(USE_FASTNOISE)
    find_package(FastNoise2 CONFIG)
    if(FastNoise2_FOUND)
        target_compile_definitions(OrganicVectoriser PUBLIC USE_FASTNOISE)
        target_link_libraries(OrganicVectoriser PUBLIC FastNoise2)
    else()
        message(WARNING "FastNoise2 not found. Perlin noise will use zero-fallback.")
    endif()
endif()

# Test executable
add_executable(vectoriser_test main.cpp)
target_link_libraries(vectoriser_test PRIVATE OrganicVectoriser)

# Install
install(TARGETS OrganicVectoriser
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} DESTINATION include/OrganicVectoriser)
