#pragma once
#include "face_binary_cls.hpp"
#include <opencv2/opencv.hpp>
#include "convolution.hpp"
#include "matrix.hpp"
#include "max_pooling.hpp"
#include "fully_connect.hpp"
#include <fstream>

bool build_network(Matrix<float> &data_im, float &result_a, float &result_b) {
    ofstream file(R"(D:\Projects\SimpleCNN\data\data.txt)", ios::out);
    Matrix<float> result_1c(64, 64, 16);
    convolution(data_im, conv_params[0], result_1c);
    for (size_t i = 0; i < 64*64*16; i++)
    {
        file << result_1c.data[i] << " ";
    }
    Matrix<float> result_1p(32, 32, 16);
    max_pooling(result_1c, result_1p);
    Matrix<float> result_2c(30, 30, 32);
    convolution(result_1p, conv_params[1], result_2c);
    Matrix<float> result_2p(15, 15, 32);
    max_pooling(result_2c, result_2p);
    Matrix<float> result_3c(8, 8, 32);
    convolution(result_2p, conv_params[2], result_3c);
    Matrix<float> result(1, 2, 1);
    fully_connect(result_3c, fc_params[0], result);
    result_a = result.data[0], result_b = result.data[1];
    return true;
}
