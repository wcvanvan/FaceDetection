#pragma once
#define _GLIBCXX_USE_CXX11_ABI 0
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
#include "cnn_network.hpp"
#include "matrix.hpp"
using namespace std;


int main() {
	cv::Mat img = cv::imread(R"(D:\Projects\SimpleCNN\pics\bg.jpg)");
	cv::resize(img, img, cv::Size(128, 128));
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);
	cv::Mat mats[3];
	cv::split(img, mats);
	float *mat_data = new float[3*128*128];
	size_t channel_size = 128 * 128;
	memcpy(mat_data, mats[0].ptr<float>(0), sizeof(float)*channel_size);
	memcpy(&mat_data[channel_size], mats[1].ptr<float>(0), sizeof(float) * channel_size);
	memcpy(&mat_data[channel_size*2], mats[2].ptr<float>(0), sizeof(float) * channel_size);
	float result_a = 0., result_b = 0.;
	if (mat_data != nullptr) {
		Matrix<float> matrix(img.rows, img.cols, img.channels(), mat_data);
		build_network(matrix, result_a, result_b);
	}
	else {
		std::cout << "No pic" << std::endl;
	}
	cout << "background score: " << result_a << ", " << "face score: " << result_b << endl;
}

