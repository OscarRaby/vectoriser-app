// fastvector.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

// --- Laplacian Smoothing (contour smoothing) ---
py::array_t<double> laplacian_smooth(py::array_t<double> contour, int iterations, double alpha) {
    auto buf = contour.request();
    if (buf.ndim != 2 || buf.shape[1] != 2)
        throw std::runtime_error("Contour must be (N,2) array");
    size_t n = buf.shape[0];
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<std::vector<double>> result(n, std::vector<double>(2));
    for (size_t i = 0; i < n; ++i) {
        result[i][0] = ptr[i*2];
        result[i][1] = ptr[i*2+1];
    }
    for (int it = 0; it < iterations; ++it) {
        std::vector<std::vector<double>> new_result = result;
        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < 2; ++j) {
                double prev = result[(i+n-1)%n][j];
                double next = result[(i+1)%n][j];
                new_result[i][j] += alpha * (prev + next - 2*result[i][j]);
            }
        }
        result = new_result;
    }
    py::array_t<double> out({n, 2});
    double* out_ptr = static_cast<double*>(out.request().ptr);
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i*2] = result[i][0];
        out_ptr[i*2+1] = result[i][1];
    }
    return out;
}

// --- Inflate Contour ---
py::array_t<double> inflate_contour(py::array_t<double> contour, double inflation_amount, double far_point_factor=1.0) {
    auto buf = contour.request();
    if (buf.ndim != 2 || buf.shape[1] != 2)
        throw std::runtime_error("Contour must be (N,2) array");
    size_t n = buf.shape[0];
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<std::vector<double>> result(n, std::vector<double>(2));
    double centroid[2] = {0, 0};
    for (size_t i = 0; i < n; ++i) {
        centroid[0] += ptr[i*2];
        centroid[1] += ptr[i*2+1];
    }
    centroid[0] /= n; centroid[1] /= n;
    std::vector<double> distances(n);
    double max_distance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = ptr[i*2] - centroid[0];
        double dy = ptr[i*2+1] - centroid[1];
        distances[i] = std::sqrt(dx*dx + dy*dy);
        if (distances[i] > max_distance) max_distance = distances[i];
    }
    for (size_t i = 0; i < n; ++i) {
        double norm_dist = (max_distance > 0) ? (distances[i] / max_distance) : 0.0;
        double scale = inflation_amount * std::exp((far_point_factor-1.0)*norm_dist);
        double dx = ptr[i*2] - centroid[0];
        double dy = ptr[i*2+1] - centroid[1];
        double norm = std::sqrt(dx*dx + dy*dy);
        double dir_x = (norm > 0) ? dx/norm : 0.0;
        double dir_y = (norm > 0) ? dy/norm : 0.0;
        result[i][0] = ptr[i*2] + dir_x * scale;
        result[i][1] = ptr[i*2+1] + dir_y * scale;
    }
    py::array_t<double> out({n, 2});
    double* out_ptr = static_cast<double*>(out.request().ptr);
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i*2] = result[i][0];
        out_ptr[i*2+1] = result[i][1];
    }
    return out;
}

// --- Module definition ---
PYBIND11_MODULE(fastvector, m) {
    m.def("laplacian_smooth", &laplacian_smooth, "Laplacian smoothing for contours");
    m.def("inflate_contour", &inflate_contour, "Inflate contour");
}
