#pragma once

#ifndef MYNEAT_UTILS_H
#define MYNEAT_UTILS_H

#include <cmath>
#include <iostream>

namespace znn {
    float Sigmoid(float x) {
        return 1.f / (1.f + std::exp(-x));
    }

    float SteependSigmoid(float x) {
        return 1.f / (1.f + std::exp(-4.9f * x));
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
        return 1.f - (result / float(outputs.size()));
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

    std::vector<std::vector<int>> ImportCSV(std::string fileName, bool hasTitle) {
        std::string line;
        std::ifstream input_file(fileName);
        uint lineCount = 0;

        if (!input_file.is_open()) {
            std::cerr << "Could not open the file - '" << fileName << "'\n";
            exit(0);
        }

        std::vector<std::vector<int>> csvDatas;

        while (getline(input_file, line)) {
            if (lineCount == 0 && hasTitle) {
                continue;
            }

            auto rawData = SplitString(line, ",");
            std::vector<int> data;

            for (auto &d : rawData) {
                data.push_back(std::stoi(d));
            }

            csvDatas.push_back(data);
            ++lineCount;
        }

        input_file.close();
        return csvDatas;
    }
}


#endif //MYNEAT_UTILS_H
