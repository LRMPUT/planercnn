//
// Created by janw on 04.01.2020.
//

// Based on https://github.com/pybind/cmake_example

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// STL
#include <random>
#include <unordered_set>
#include <iostream>

// Eigen
#include <Eigen/Dense>

namespace py = pybind11;

std::vector<py::array_t<float>> comp_score(py::array_t<float> points, int numIter, float planeDiffThres) {
    // static constexpr int numIter = 10;
    // static constexpr double planeDiffThres = 0.01;

    int nanchors = points.shape(0);
    int maskH = points.shape(2);
    int maskW = points.shape(3);

    py::array_t<float> retScores(py::array::ShapeContainer{nanchors});
    py::array_t<float> retMasks(py::array::ShapeContainer{nanchors, maskH, maskW});
    py::array_t<float> retPlanes(py::array::ShapeContainer{nanchors, 3});

    for(int a = 0; a < nanchors; ++a){
        retScores.mutable_at(a) = 0.0;
        for(int r = 0; r < maskH; ++r) {
            for (int c = 0; c < maskW; ++c) {
                retMasks.mutable_at(a, r, c) = 0.0;
            }
        }

        std::vector<std::pair<int, int>> validCoords;
        Eigen::MatrixXf validPoints(maskH * maskW, 3);
        int nvalid = 0;
        for(int r = 0; r < maskH; ++r) {
            for(int c = 0; c < maskW; ++c) {
                Eigen::Vector3f pt(points.at(a, 0, r, c),
                                   points.at(a, 1, r, c),
                                   points.at(a, 2, r, c));

                if(pt.norm() > 1.0e-3f) {
                    validCoords.push_back(std::make_pair(r, c));
                    validPoints.block<1, 3>(nvalid, 0) = pt.transpose();

                    ++nvalid;
                }
            }
        }
        validPoints.conservativeResize(nvalid, 3);

        if(validCoords.size() >= 3) {
            // RANSAC
            std::random_device rd;
            std::mt19937 gen(rd());

            int bestNumInliers = 0;
            std::vector<int> bestInliers;
            for (int i = 0; i < numIter; ++i) {
                std::uniform_int_distribution<> dis(0, validCoords.size() - 1);
                std::unordered_set<int> curIdxs;

                Eigen::Matrix3f A;
                int npts = 0;
                while (curIdxs.size() < 3) {
                    int curIdx = dis(gen);

                    if(curIdxs.insert(curIdx).second) {
                        A.block<1, 3>(npts, 0) = validPoints.block<1, 3>(curIdx, 0);

                        ++npts;
                    }
                }
                if(std::abs(A.determinant()) > 1e-3) {
                    Eigen::Vector3f plane = A.partialPivLu().solve(Eigen::Vector3f::Ones());

                    Eigen::MatrixXf diff = (validPoints * plane - Eigen::MatrixXf::Ones(nvalid, 1)).cwiseAbs();
                    std::vector<int> curInliers;
                    for(int p = 0; p < nvalid; ++p) {
                        if(diff(p) < planeDiffThres) {
                            curInliers.push_back(p);
                        }
                    }
                    if(curInliers.size() > bestInliers.size()) {
                        curInliers.swap(bestInliers);
                    }
                }
            }

            if(bestInliers.size() >= 3) {
                Eigen::MatrixXf inlierPoints(bestInliers.size(), 3);
                for(int p = 0; p < bestInliers.size(); ++p) {
                    inlierPoints.block<1, 3>(p, 0) = validPoints.block<1, 3>(bestInliers[p], 0);
                }
                Eigen::Vector3f plane = inlierPoints.householderQr().solve(Eigen::MatrixXf::Ones(bestInliers.size(), 1));

                retScores.mutable_at(a) = (float)bestInliers.size() / (maskH * maskW);

                for(int p = 0; p < bestInliers.size(); ++p) {
                    int idx = bestInliers[p];
                    retMasks.mutable_at(a, validCoords[idx].first, validCoords[idx].second) = 1.0;
                }

                retPlanes.mutable_at(a, 0) = plane(0);
                retPlanes.mutable_at(a, 1) = plane(1);
                retPlanes.mutable_at(a, 2) = plane(2);
            }
        }
    }

    return {retScores, retMasks, retPlanes};
}

PYBIND11_MODULE(comp_score_py, m) {
    m.doc() = R"pbdoc(
            Python bindings for comp score
            -----------------------
            .. currentmodule:: comp_score_py
            .. autosummary::
               :toctree: _generate
               comp_score
        )pbdoc";

    m.def("comp_score",
            &comp_score,
            py::arg("points"),
            py::arg("num_iter"),
            py::arg("plane_diff_thres"),
            R"pbdoc(
            Compute score.
            Compute score.
        )pbdoc");

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}
