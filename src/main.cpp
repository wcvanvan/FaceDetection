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
	//float* data = new float[8] { 1,2,3,4,5,6,7,8 };
	//Matrix<float> mat(2, 4, 1, data);
	//Matrix<float> mat_r(1, 2, 1);
	//convolution(mat, conv_params[3], mat_r);
	//cout << mat_r.data[0] << endl;

	//float numbers[27];
	////for (size_t i = 0; i < 3; i++)
	////{
	////	numbers[i * 4 + 0] = mat_data[128 * 128 * i + 0];
	////	numbers[i * 4 + 1] = mat_data[128 * 128 * i + 1];
	////	numbers[i * 4 + 2] = mat_data[128 * 128 * i + 128];
	////	numbers[i * 4 + 3] = mat_data[128 * 128 * i + 129];		

	////}
	//for (size_t i = 0; i < 27; i++)
	//{
	//	numbers[i] = conv0_weight[i];
	//}
	//ofstream file(R"(D:\Projects\SimpleCNN\data\data_weight.txt)");
	//for (size_t i = 0; i < 27; i++)
	//{
	//	if (i % 9 == 0) {
	//		file << endl;
	//	}
	//	file << numbers[i] << ", ";
	//}
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
	cout << result_a << ", " << result_b << endl;
}

