#pragma once

#ifndef MYNEAT_UTILS_H
#define MYNEAT_UTILS_H

#include <cmath>
#include <iostream>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <utility>
#include <random>
#include <fstream>

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

    // [FIX] 原实现是 (4.9*(1-f))^2，不是 SteependSigmoid 的导数；
    // 正确的 f'(x) = 4.9*f(x)*(1-f(x))，按"以输出值求导"的约定实现为 4.9*fx*(1-fx)
    float DerivativeSteependSigmoid(float fx) {
        return 4.9f * fx * (1.f - fx);
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

    // [OPT] 三个精度函数改为 const 引用：热路径（每样本×每个体×每代）按值传参会产生整份向量拷贝
    float AbsoluteDeviation(const std::vector<float> &outputs, const std::vector<float> &wantedOutputs) {
        float result = 0.f;
        for (uint i = 0; i < outputs.size(); ++i) {
            result += std::abs(wantedOutputs[i] - outputs[i]);
        }
        return result / float(outputs.size());
    }

    float Variance(const std::vector<float> &outputs, const std::vector<float> &wantedOutputs) {
        float result = 0.f;
        for (uint i = 0; i < outputs.size(); ++i) {
            result += std::pow(wantedOutputs[i] - outputs[i], 2.f);
        }
        return result / float(outputs.size());
    }

    float StandardDeviation(const std::vector<float> &outputs, const std::vector<float> &wantedOutputs) {  // 如果输出节点数量为1，则标准差和绝对差的计算结果一样，应使用绝对差减少运算量
        return std::sqrt(Variance(outputs, wantedOutputs));
    }

    // [OPT] MinMax 归一化：二维数据按列（第一维=样本，第二维=特征）线性缩放到 [min,max]，默认 [0,1]。
    // sigmoid/steepend-sigmoid 在原始大数值输入（如 iris 的 4~8、像素 0~255）下饱和，梯度接近 0，
    // BP 收敛极慢（iris 基准 605ms 主要耗在饱和区）。归一化后梯度恢复，训练可快一个数量级。
    // 某列 max==min（常数列）时该列输出 0。app 端决定是否调用：训练与部署需用同一变换。
    inline std::vector<std::vector<float>> NormalizeData(const std::vector<std::vector<float>> &data, float min = 0.f, float max = 1.f) {
        if (data.empty() || data[0].empty()) {
            return data;
        }

        std::vector<float> colMin = data[0], colMax = data[0];
        for (auto &row: data) {
            for (uint c = 0; c < row.size() && c < colMin.size(); ++c) {
                if (row[c] < colMin[c]) {
                    colMin[c] = row[c];
                }
                if (row[c] > colMax[c]) {
                    colMax[c] = row[c];
                }
            }
        }

        std::vector<std::vector<float>> result;
        result.reserve(data.size());
        for (auto &row: data) {
            std::vector<float> newRow(row.size());
            for (uint c = 0; c < row.size(); ++c) {
                float span = colMax[c] - colMin[c];
                newRow[c] = span > 0.f ? min + (row[c] - colMin[c]) / span * (max - min) : min;
            }
            result.push_back(newRow);
        }
        return result;
    }

    // [FIX] 原实现在字符串以分隔符结尾时会追加一个空 token（std::stof("") 会抛异常），
    // 且直接用 data()+size() 构造在 delimiter 为空串时行为异常；改为标准 find 切分并跳过空 token
    std::vector<std::string> SplitString(std::string &target, std::string delimiter) {
        std::vector<std::string> vs;
        size_t pos = 0;
        while (pos <= target.size()) {
            size_t fd = target.find(delimiter, pos);
            if (fd == std::string::npos) {
                std::string token = target.substr(pos);
                if (!token.empty()) {
                    vs.push_back(token);
                }
                break;
            }
            std::string token = target.substr(pos, fd - pos);
            if (!token.empty()) {
                vs.push_back(token);
            }
            pos = fd + delimiter.size();
        }
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

            for (auto &d: rawData) {
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

    // [FIX] 原实现裸 new[] 返回裸指针，调用方异常路径或忘记 delete 会泄漏；改为 std::vector 自动管理内存
    // 注意：返回类型从 uint* 改为 std::vector<uint>，现有调用方均未使用此函数（grep 全库无调用）
    inline std::vector<uint> GetRandIndex(uint arrayLength) {
        std::vector<uint> arr(arrayLength);
        for (uint i = 0; i < arrayLength; ++i) {
            arr[i] = i;
        }

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(arr.begin(), arr.end(), g);

        return arr;
    }
}


#endif //MYNEAT_UTILS_H
