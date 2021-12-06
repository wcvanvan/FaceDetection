#pragma once
#include "face_binary_cls.hpp"
#include <opencv2/opencv.hpp>
#include "convolution.hpp"
#include "matrix.hpp"

class CNNNetwork {
    float final_result[2] = {0.,0.};
    explicit CNNNetwork(Matrix &data_im);f
    void build(Matrix &data_im);
};

CNNNetwork::CNNNetwork(Matrix &data_im) {
    build(mat);
}

void CNNNetwork::build(Matrix &data_im) {
    Matrix<float> result_a(64, 64, 16);
    convolution(data_im, conv_params[0], result_a);
    Matrix<float> result_b(62, 62, 32);
    convolution(result_a, conv_params[1], result_b);
    Matrix<float> result_c(31, 31, 32);
    convolution(result_b, conv_params[2], result_c);
    Matrix<float> result(1,2,1);
    convolution(result_c, fc_params[0], result);
    for (int i = 0; i < 2; ++i) {
        final_result[i] = result.data[i];
    }
}
