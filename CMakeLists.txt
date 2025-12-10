cmake_minimum_required(VERSION 3.4...3.18)
project(fastvector LANGUAGES CXX)

# Find pybind11 (make sure it's installed: pip install pybind11)
find_package(pybind11 REQUIRED)

# Add your C++ source file(s) here
pybind11_add_module(fastvector fastvector.cpp)

# Optional: set C++ standard
set_target_properties(fastvector PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)