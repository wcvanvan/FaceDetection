#pragma once

#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include "mat_mul.hpp"

using namespace std;

template<typename T>
class Matrix {
public:
    size_t rows{}, cols{};
    int channels{};
    int *ref_count{nullptr};
    T *data{nullptr};
    T *data_start;
    int row_start{}, row_end{}, col_start{}, col_end{};
    int rowsROI{}, colsROI{};

    Matrix<T>() = default;

    Matrix<T>(size_t _rows, size_t _cols, int _channels, T *_data = nullptr);

    Matrix<T>(Matrix &m);

    ~Matrix();

    void create(T *_data = nullptr);

    T *at(int row, int col);

    T *ptr(int row);

    void locateROI(int _row_start, int _row_end, int _col_start, int _col_end);

    void adjustROI(int d_top, int d_bottom, int d_left, int d_right);

    Matrix &operator=(const Matrix &m);

    bool operator==(const Matrix &m);

    Matrix &operator+(const Matrix &m);

    Matrix &operator+(T var);

    template<typename U>
    friend Matrix &operator+(U var, const Matrix &m);

    Matrix &operator-(const Matrix &m);

    Matrix &operator-(T var);

    template<typename U>
    friend Matrix &operator-(U var, const Matrix &m);

    Matrix &operator*(Matrix &m);

    Matrix &operator*(int scalar);

    void release();
};

template<typename T>
Matrix<T>::Matrix(size_t _rows, size_t _cols, int _channels, T *_data) {
    rows = _rows;
    cols = _cols;
    row_start = 0, row_end = rows - 1;
    col_start = 0, col_end = cols - 1;
    channels = _channels;
    rowsROI = rows, colsROI = cols;
    create(_data);
}

template<typename T>
Matrix<T>::Matrix(Matrix &m) {
    *this = m;
}

template<typename T>
Matrix<T>::~Matrix() {
    release();
}

template<typename T>
void Matrix<T>::create(T *_data) {
    if (data != nullptr) {
        return;
    }
    if (_data == nullptr) {
        data = new T[rows * cols * channels];
    } else {
        data = _data;
    }
    ref_count = new int{1};
    data_start = data;
}

template<typename T>
T *Matrix<T>::at(int row, int col) {
    return data_start + row * cols * channels + col * channels;
}

template<typename T>
T *Matrix<T>::ptr(int row) {
    return data_start + row * cols * channels;
}

template<typename T>
void Matrix<T>::locateROI(int _row_start, int _row_end, int _col_start,
                          int _col_end) {
    row_start = (_row_start > 0) ? _row_start : 0;
    row_end = (_row_end < rows - 1) ? _row_end : rows - 1;
    col_start = (_col_start > 0) ? _col_start : 0;
    col_end = (_col_end < cols - 1) ? _col_end : cols - 1;
    data_start = data + (row_start * cols + col_start) * channels;
    rowsROI = row_end - row_start + 1;
    colsROI = col_end - col_start + 1;
}

template<typename T>
void Matrix<T>::adjustROI(int d_top, int d_bottom, int d_left, int d_right) {
    d_top = (d_top < row_start) ? d_top : row_start;
    d_bottom = (d_bottom < rows - 1 - row_end) ? d_bottom : rows - 1 - row_end;
    d_left = (d_left < col_start) ? d_left : col_start;
    d_right = (d_right < cols - 1 - col_end) ? d_right : cols - 1 - col_end;
    data_start -= (d_top * cols * channels + d_left * channels);
    rowsROI += d_top + d_bottom;
    colsROI += d_left + d_right;
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix &m) {
    if (this == &m) {
        return *this;
    }
    if (this->data != nullptr) {
        // 释放原始数据
    }
    this->rows = m.rows;
    this->cols = m.cols;
    this->channels = m.channels;
    this->ref_count = m.ref_count;
    (*(this->ref_count))++;
    this->data = m.data;
    this->data_start = m.data_start;
    this->rowsROI = m.rowsROI;
    this->colsROI = m.colsROI;
    this->row_start = m.row_start;
    this->row_end = m.row_end;
    this->col_start = m.col_start;
    this->col_end = m.col_end;
    return *this;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix &m) {
    if (this->row_start != m.row_start || this->col_start != m.col_start ||
        this->row_end == m.row_end || this->col_end == m.col_end ||
        this->channels != m.channels) {
        return false;
    }
    if (this->ref_count != m.ref_count || this->data != m.data ||
        this->data_start != m.data_start) {
        return false;
    }
    return true;
}

template<typename T>
Matrix<T> &Matrix<T>::operator+(const Matrix<T> &m) {
    if (this->rowsROI != m.rowsROI || this->colsROI != m.colsROI ||
        this->channels != m.channels) {
        std::cerr << "operands have different types" << std::endl;
        return *this;
    }
    auto *mat = new Matrix(this->rowsROI, this->colsROI, this->channels);
    auto ptr_a = this->data_start;
    auto ptr_b = m.data_start;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            for (int k = 0; k < mat->channels; k++) {
                mat->data[i * mat->cols * mat->channels + j * mat->channels + k] =
                        ptr_a[i * this->cols * this->channels + j * this->channels + k] +
                        ptr_b[i * m.cols * m.channels + j * m.channels + k];
            }
        }
    }
    return *mat;
}

template<typename T>
Matrix<T> &Matrix<T>::operator+(T var) {
    auto *mat = new Matrix(this->rowsROI, this->colsROI, this->channels);
    auto ptr_a = this->data_start;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            for (int k = 0; k < mat->channels; k++) {
                mat->data[i * mat->cols * mat->channels + j * mat->channels + k] =
                        ptr_a[i * this->cols * this->channels + j * this->channels + k] +
                        var;
            }
        }
    }
    return *mat;
}

template<typename T>
Matrix<T> &operator+(T var, Matrix<T> &m) {
    return m + var;
}

template<typename T>
Matrix<T> &Matrix<T>::operator-(const Matrix<T> &m) {
    if (this->rowsROI != m.rowsROI || this->colsROI != m.colsROI ||
        this->channels != m.channels) {
        std::cerr << "operands have different types" << std::endl;
        return *this;
    }
    auto *mat = new Matrix(this->rowsROI, this->colsROI, this->channels);
    auto ptr_a = this->data_start;
    auto ptr_b = m.data_start;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            for (int k = 0; k < mat->channels; k++) {
                mat->data[i * mat->cols * mat->channels + j * mat->channels + k] =
                        ptr_a[i * this->cols * this->channels + j * this->channels + k] -
                        ptr_b[i * m.cols * m.channels + j * m.channels + k];
            }
        }
    }
    return *mat;
}

template<typename T>
Matrix<T> &Matrix<T>::operator-(T var) {
    if (!(typeid(this->ptr[0]) == typeid(T))) {
        std::cerr << "incompatible type" << std::endl;
        return *this;
    }
    auto *mat = new Matrix(this->rowsROI, this->colsROI, this->channels);
    auto ptr_a = this->data_start;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            for (int k = 0; k < mat->channels; k++) {
                mat->data[i * mat->cols * mat->channels + j * mat->channels + k] =
                        ptr_a[i * this->cols * this->channels + j * this->channels + k] -
                        var;
            }
        }
    }
    return *mat;
}

template<typename T>
Matrix<T> &operator+(T var, const Matrix<T> &m) {
    return m + var;
}

template<typename T>
Matrix<T> &operator-(T var, const Matrix<T> &m) {
    auto *mat = new Matrix<T>(m.rowsROI, m.colsROI, m.channels);
    auto ptr_a = m.data_start;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            for (int k = 0; k < mat->channels; k++) {
                mat->data[i * mat->cols * mat->channels + j * mat->channels + k] =
                        var - ptr_a[i * m.cols * m.channels + j * m.channels + k];
            }
        }
    }
    return *mat;
}

template<typename T>
Matrix<T> &Matrix<T>::operator*(Matrix &m) {
    if (this->colsROI != m.rowsROI || this->channels != 1 || m.channels != 1) {
        std::cerr << "not suitable matrix to do multiplication" << std::endl;
    }
    auto *mat = new Matrix<T>(m.rowsROI, m.colsROI, m.channels);
    if (this->rowsROI == 1) {
        mul_conv(this->data_start, m.data_start, mat->data_start, this->rowsROI, m.colsROI, this->colsROI);
    }
    else if (m.rowsROI >= 128) {
        // dividing the whole matrix into pow(bigBlockCountInRow, 2)'s big blocks.
        // Each big block contains 64 (8x8) small blocks of size 8x8
        int bigBlockCountInRow = m.rowsROI / 64;
        for (int z = 0; z < pow(bigBlockCountInRow, 2); z++) {
            int newI = z / bigBlockCountInRow * 64;
            int newJ = z % bigBlockCountInRow * 64;
            for (int i = newI; i < newI + 64; i += 8) {
                for (int j = newJ; j < newJ + 64; j += 8) {
                    addDot_8x8(&this->data_start[i * this->cols], &m.data_start[j],
                               &mat->data[i * mat->cols + j], this->rowsROI, m.colsROI, this->colsROI);
                }
            }
        }
    } else if (m.rowsROI >= 16) {
        int bigBlockCountInRow = m.rowsROI / 16;
        // dividing the whole matrix into pow(bigBlockCountInRow, 2)'s big blocks.
        // Each big block contains 16 (4x4) small blocks of size 4x4
        for (int z = 0; z < pow(bigBlockCountInRow, 2); z++) {
            int newI = z / bigBlockCountInRow * 16;
            int newJ = z % bigBlockCountInRow * 16;
            for (int i = newI; i < newI + 16; i += 4) {
                for (int j = newJ; j < newJ + 16; j += 4) {
                    addDot_4x4(&this->data_start[i * this->cols], &m.data_start[j], &mat->data[i * mat->cols + j],
                               this->rowsROI, m.colsROI, this->colsROI);
                }
            }
        }
    } else {
        for (int i = 0; i < this->rowsROI; i++) {
            for (int k = 0; k < this->colsROI; k++) {
                for (int j = 0; j < m.colsROI; j++) {
                    mat->data_start[i * mat->cols + j] +=
                            this->data_start[i + this->colsROI + k] * m.data_start[k * m.colsROI + j];
                }
            }
        }
    }
    return *mat;
}

template<typename T>
Matrix<T> &Matrix<T>::operator*(int scalar) {
    auto *mat = new Matrix<T>(this->rowsROI, this->colsROI, this->channels);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            for (int k = 0; k < mat->channels; k++) {
                mat->data[i * mat->cols * mat->channels + j * mat->channels + k] =
                        this->data_start[i * this->cols * this->channels + j * this->channels + k] * scalar;
//                cout << this->data_start[i * this->cols * this->channels + j * this->channels + k] << " " << scalar << endl;
            }
        }
    }
    return *mat;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, Matrix<T> &m) {
    auto ptr = m.data;
    for (int i = m.row_start; i <= m.row_end; i++) {
        for (int j = m.col_start; j <= m.col_end; j++) {
            os << "{ ";
            for (int k = 0; k < m.channels; k++) {
                os << ptr[i * m.cols * m.channels + j * m.channels + k];
                if (k != m.channels - 1) {
                    os << ", ";
                }
            }
            os << " }";
        }
        os << endl;
    }
    return os;
}

template<typename T>
std::istream &operator>>(std::istream &is, Matrix<T> &m) {
    auto ptr = m.data_start;
    for (int i = m.row_start; i <= m.row_end; i++) {
        for (int j = m.col_start; j <= m.col_end; j++) {
            for (int k = 0; k < m.channels; k++) {
                is >> ptr[i * m.cols * m.channels + j * m.channels + k];
            }
        }
    }
    return is;
}

template<typename T>
void Matrix<T>::release() {
    cout << "ref_count-1" << endl;
    if (*ref_count <= 1) {
        delete ref_count;
        delete data;
        cout << "data is released" << endl;
        return;
    }
    (*ref_count)--;
}
