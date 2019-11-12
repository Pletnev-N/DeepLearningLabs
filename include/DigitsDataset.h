#pragma once

#include <cv.h>
#include <cxcore.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

class DigitsDataset
{
public:
    DigitsDataset(std::string path) :
        train_images_file_name("train-images.idx3-ubyte"),
        train_labels_file_name("train-labels.idx1-ubyte"),
        test_images_file_name("t10k-images.idx3-ubyte"),
        test_labels_file_name("t10k-labels.idx1-ubyte")
    {
        train_images_file.open(path + "\\" + train_images_file_name, std::ios::in | std::ios::binary);
        train_labels_file.open(path + "\\" + train_labels_file_name, std::ios::in | std::ios::binary);
        test_images_file.open(path + "\\" + test_images_file_name, std::ios::in | std::ios::binary);
        test_labels_file.open(path + "\\" + test_labels_file_name, std::ios::in | std::ios::binary);

        reset();

        n_train = 60000;
        n_test = 10000;

        rows = 28;
        cols = 28;
    }

    ~DigitsDataset()
    {
        train_images_file.close();
        train_labels_file.close();
        test_images_file.close();
        test_labels_file.close();
    }

    void reset()
    {
        train_images_file.seekg(16);
        train_labels_file.seekg(8);
        test_images_file.seekg(16);
        test_labels_file.seekg(8);
    }

    void get_next_train(cv::Mat & image, int & label)
    {
        label = 0;
        train_labels_file.read((char*)(&label), 1);

        get_next_image(train_images_file, image);
    }

    void get_next_test(cv::Mat & image, int & label)
    {
        label = 0;
        test_labels_file.read((char*)(&label), 1);

        get_next_image(test_images_file, image);
    }

    void get_next_train_col(cv::Mat & image, int & label)
    {
        label = 0;
        train_labels_file.read((char*)(&label), 1);

        get_next_image_col(train_images_file, image);
    }

    void get_next_test_col(cv::Mat & image, int & label)
    {
        label = 0;
        test_labels_file.read((char*)(&label), 1);

        get_next_image_col(test_images_file, image);
    }

    int get_n_train()
    {
        return n_train;
    }

    int get_n_test()
    {
        return n_test;
    }

private:
    void get_next_image(std::ifstream & stream, cv::Mat & image)
    {
        image.create(rows, cols, CV_8UC1);
        char pixel;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                stream.read(&pixel, 1);
                image.at<uchar>(i, j) = pixel;
            }

        image.convertTo(image, CV_64FC1);
        image /= 255;
    }

    void get_next_image_col(std::ifstream & stream, cv::Mat & image)
    {
        image.create(rows * cols, 1, CV_8UC1);
        char pixel;

        for (int i = 0; i < rows * cols; i++)
        {
            stream.read(&pixel, 1);
            image.at<uchar>(i, 0) = pixel;
        }

        image.convertTo(image, CV_64FC1);
        image /= 255;
    }

    std::string const train_images_file_name;
    std::string const train_labels_file_name;
    std::string const test_images_file_name;
    std::string const test_labels_file_name;

    std::ifstream train_images_file;
    std::ifstream train_labels_file;
    std::ifstream test_images_file;
    std::ifstream test_labels_file;

    int n_train;
    int n_test;
    int rows;
    int cols;
};
