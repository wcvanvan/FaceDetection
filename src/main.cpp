#pragma once
#define _GLIBCXX_USE_CXX11_ABI 0
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cnn_network.hpp"
#include "matrix.hpp"
using namespace std;


int main() {
	cv::Mat img = cv::imread(R"(D:\Projects\SimpleCNN\pics\face.jpg)");
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);
	float* mat_data = img.ptr<float>(0);
	float result_a = 0., result_b = 0.;
	if (mat_data != nullptr) {
		Matrix<float> matrix(img.rows, img.cols, img.channels(), mat_data);
		matrix.isStaticData = true;
		build_network(matrix, result_a, result_b);
	}
	else {
		std::cout << "No pic" << std::endl;
	}
	cout << "background score: " << result_a << ", " << "face score: " << result_b << endl;
}

