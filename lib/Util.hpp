#pragma once

#ifndef MYNEAT_UTILS_H
#define MYNEAT_UTILS_H

#include <cmath>
#include <iostream>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <utility>

namespace znn {
    float Tanh(float x) {
//        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
        return 2.f / (1.f + std::exp(-2.f * x)) - 1.f;
    }

    float DerivativeTanh(float fx) {
        return 1.f - std::pow(fx, 2.f);
    }

    float Sigmoid(float x) {
        return 1.f / (1.f + std::exp(-x));
    }

    float DerivativeSigmoid(float fx) {
        return fx * (1.f - fx);
    }

    float SteependSigmoid(float x) {
        return 1.f / (1.f + std::exp(-4.9f * x));
    }

    float DerivativeSteependSigmoid(float fx) {
        float s = std::exp(-4.9f * fx);
        return std::pow(s * 4.9f / (1.f + s), 2.f);
    }

    float MySteependSigmoid(float x) {
        return 1.f / (1.f + std::exp(-9.9f * x));
    }

    float MyGentleSigmoid(float x) {
        return 1.f / (1.f + std::exp(-x / 3.f));
    }

    float MyWtfSigmoid(float x) {
        if (x < 0.f) {
            return 0.f;
        }
        if (x > 0.f) {
            return 1.f;
        }
        return 0.5f;
    }

    float StandardDeviation(std::vector<float> outputs, std::vector<float> wantedOutputs) {
        float result = 0.f;
        for (uint i = 0; i < outputs.size(); ++i) {
            result += std::pow(wantedOutputs[i] - outputs[i], 2.f);
        }
        return result / float(outputs.size());
    }

    float GetPrecision(std::vector<float> outputs, std::vector<float> wantedOutputs) {
        return 1.f - StandardDeviation(outputs, wantedOutputs);
    }

    std::vector<std::string> SplitString(std::string &target, std::string delimiter) {
        std::vector<std::string> vs;
        size_t pos{};
//        for (size_t fd = 0; (fd = target.find(delimiter, pos)) != std::string::npos; pos = fd + delimiter.size()) {
        for (size_t fd = 0; (fd = target.find(delimiter, pos)) != std::string::npos; pos = fd + delimiter.size()) {
            vs.emplace_back(target.data() + pos, target.data() + fd);
        }
        vs.emplace_back(target.data() + pos, target.data() + target.size());
        return vs;
    }

    std::vector<std::vector<float>> ImportCSV(std::string fileName, bool hasTitle) {
        std::string line;
        std::ifstream input_file(fileName);
        uint lineCount = 0;

        if (!input_file.is_open()) {
            std::cerr << "Could not open the file - '" << fileName << "'\n";
            exit(0);
        }

        std::vector<std::vector<float>> csvDatas;

        while (getline(input_file, line)) {
            if (lineCount == 0 && hasTitle) {
                continue;
            }

            auto rawData = SplitString(line, ",");
            std::vector<float> data;

            for (auto &d : rawData) {
                data.push_back(std::stof(d));
            }

            csvDatas.push_back(data);
            ++lineCount;
        }

        input_file.close();
        return csvDatas;
    }

    template<typename T>
    std::vector<size_t> SortIndexes(const std::vector<T> &v) {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
            return v[i1] > v[i2];
        });

        return idx;
    }
}


#endif //MYNEAT_UTILS_H
