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
#include <queue>

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

py::tuple comp_components(py::array_t<int> segments,
                          //py::array_t<float> planes,
                          //py::array_t<float> planeInfo,
                          int segmentAreaThresh,
                          float segmentAreaRatio)
{
    py::array_t<int> retComp(py::array::ShapeContainer{segments.shape(0), segments.shape(1)});
    for(int r = 0; r < segments.shape(0); ++r){
        for(int c = 0; c < segments.shape(1); ++c){
            retComp.mutable_at(r, c) = -1;
        }
    }
    int nextId = 0;
    std::vector<int> newIdToOldId;
    for(int r = 0; r < segments.shape(0); ++r){
        for(int c = 0; c < segments.shape(1); ++c){
            if(retComp.at(r, c) == -1){
                std::queue<std::pair<int, int>> q;
                q.push(std::make_pair(r, c));

                int area = 1;
                int minx = c;
                int maxx = c;
                int miny = r;
                int maxy = r;
                int oldId = segments.at(r, c);
                while(!q.empty()){
                    int curr = q.front().first;
                    int curc = q.front().second;
                    q.pop();
                    retComp.mutable_at(curr, curc) = nextId;

                    int nh[][2] = {{-1, 0},
                                   {1, 0},
                                   {0, -1},
                                   {0, 1}};
                    for(int ni = 0; ni < sizeof(nh)/sizeof(nh[0]); ++ni){
                        int nhr = curr + nh[ni][0];
                        int nhc = curc + nh[ni][1];
                        if(nhr >= 0 && nhr < segments.shape(0) && nhc >= 0 && nhc < segments.shape(1)){
                            if(segments.at(nhr, nhc) == oldId &&
                                retComp.at(nhr, nhc) != nextId &&
                                (retComp.at(nhr, nhc) == -1 || newIdToOldId[retComp.at(nhr, nhc)] == -1))
                            {
                                int narea = area + 1;
                                int nminx = std::min(minx, nhc);
                                int nmaxx = std::max(maxx, nhc);
                                int nlenx = nmaxx - nminx + 1;
                                int nminy = std::min(miny, nhr);
                                int nmaxy = std::max(maxy, nhr);
                                int nleny = nmaxy - nminy + 1;
                                if(nlenx > nleny){
                                    nleny = std::max(nleny, nlenx / 2);
                                }
                                else{
                                    nlenx = std::max(nlenx, nleny / 2);
                                }
                                int nboxarea = nlenx * nleny;
                                if(narea < 100 || narea > nboxarea * segmentAreaRatio){
                                    retComp.mutable_at(nhr, nhc) = nextId;
                                    q.push(std::make_pair(nhr, nhc));

                                    area = narea;
                                    minx = nminx;
                                    maxx = nmaxx;
                                    miny = nminy;
                                    maxy = nmaxy;
                                }
                            }
                        }
                    }
                }
                if(area > segmentAreaThresh){

                    newIdToOldId.push_back(oldId);
                }
                else{
                    newIdToOldId.push_back(-1);
                }
                ++nextId;
            }
        }
    }

    // py::array_t<float> retPlanes(py::array::ShapeContainer{(long)newIdToOldId.size(), planes.shape(1)});
    // py::array_t<float> retPlaneInfo(py::array::ShapeContainer{(long)newIdToOldId.size(), planeInfo.shape(1)});
    // for(int i = 0; i < newIdToOldId.size(); ++i){
    //     if(newIdToOldId[i] >= 0){
    //         for(int v = 0; v < planes.shape(1); ++v){
    //             retPlanes.mutable_at(i, v) = planes.at(newIdToOldId[i], v);
    //         }
    //         for(int v = 0; v < planeInfo.shape(1); ++v){
    //             retPlaneInfo.mutable_at(i, v) = planeInfo.at(newIdToOldId[i], v);
    //         }
    //     }
    //     else{
    //         for(int v = 0; v < planes.shape(1); ++v){
    //             retPlanes.mutable_at(i, v) = 0.0f;
    //         }
    //         for(int v = 0; v < planeInfo.shape(1); ++v){
    //             retPlaneInfo.mutable_at(i, v) = 0.0f;
    //         }
    //     }
    // }
    // return py::make_tuple(retComp, retPlanes, retPlaneInfo);

    py::array_t<int> retNewIdToOldId(py::array::ShapeContainer{(long)newIdToOldId.size()});
    for(int i = 0; i < newIdToOldId.size(); ++i){
        retNewIdToOldId.mutable_at(i) = newIdToOldId[i];
    }
    return py::make_tuple(retComp, retNewIdToOldId);
}


PYBIND11_MODULE(comp_score_py, m) {
    m.doc() = R"pbdoc(
            Python bindings for comp score
            -----------------------
            .. currentmodule:: comp_score_py
            .. autosummary::
               :toctree: _generate
               comp_score
               comp_componenets
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

    m.def("comp_components",
          &comp_components,
          py::arg("segments"),
          //py::arg("planes"),
          //py::arg("planes_info"),
          py::arg("segment_area_threshold"),
          py::arg("segment_area_ratio"),
          R"pbdoc(
            Compute components.
            Compute components.
        )pbdoc");

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}
