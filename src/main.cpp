#pragma once

#include <cstring>
#include <iostream>
#include "mat_mul.hpp"
#include "face_binary_cls.hpp"
#include "convolution.hpp"

using namespace std;
#include <opencv2/opencv.hpp>
#include "matrix.hpp"

int main() {
    cv::Mat mat = cv::imread("/mnt/d/Projects/CNN/pics/face.jpg");
    cv::Mat img;
    mat.convertTo(img, CV_32FC3, 1.0/255.0);
    if (img.data != nullptr) {
        std::cout << typeid(img.data).name() << std::endl;
//        Matrix<float> matrix(img.rows, img.cols, img.channels(), reinterpret_cast<float *>(img.data));
    } else {
        std::cout << "No pic" << std::endl;
    }
}

//#include <stdio.h>
//
//#include <iostream>
//
//using namespace std;
//
//int main(int argc, char **argv) {
//
//    int no_os_flag = 1;
//
//#ifdef linux
//
//    no_os_flag = 0;
//
//  cout << "It is in Linux OS!" << endl;
//
//#endif
//
//#ifdef _UNIX
//
//    no_os_flag = 0;
//
//  cout << "It is in UNIX OS!" << endl;
//
//#endif
//
//#ifdef __WINDOWS_
//
//    no_os_flag = 0;
//
//  cout << "It is in Windows OS!" << endl;
//
//#endif
//
//#ifdef _WIN32
//
//    no_os_flag = 0;
//
//    cout << "It is in WIN32 OS!" << endl;
//
//#endif
//
//    if (1 == no_os_flag) {
//
//        cout << "No OS Defined ,I do not know what the os is!" << endl;
//    }
//
//    return 0;
//}
