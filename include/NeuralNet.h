#pragma once

#include <cv.h>
#include <cxcore.h>
#include <stdlib.h>
#include <iostream>

#include "DigitsDataset.h"

class NeuralNet
{
public:
    NeuralNet(int input_layer_size, int hidden_layer_size, int output_layer_size) :
        N(input_layer_size),
        K(hidden_layer_size),
        M(output_layer_size)
    {
        w1.create(K, N, CV_64FC1);
        w2.create(M, K, CV_64FC1);

        double low = -1.0;
        double high = 1.0;
        cv::randu(w1, cv::Scalar(low), cv::Scalar(high));
        cv::randu(w2, cv::Scalar(low), cv::Scalar(high));
    }

    cv::Mat get_output(cv::Mat input)
    {
        cv::Mat res(M, 1, CV_64FC1);

        //hidden layer output
        res = logistic(w1 * input);

        // net output
        res = softmax(w2 * res);

        return res;
    }

    void train(DigitsDataset & data, int iter_max, double alpha)
    {
        int iter = 0;
        int n_train = data.get_n_train();

        cv::Mat x, y;
        int label;

        while (iter < iter_max)
        {
            if (iter % n_train == 0) data.reset();
            iter++;

            if (iter % 1000 == 0) std::cout << "i " << iter << std::endl;

            data.get_next_train_col(x, label);
            y = create_y(label);
            train_iteration(x, y, label, alpha);
        }
    }

    double error_test(DigitsDataset & data)
    {
        double error = 0;

        int n_test = data.get_n_test();

        cv::Mat x, y;
        int label;

        data.reset();
        for (int i = 0; i < n_test; i++)
        {
            data.get_next_test_col(x, label);
            y = get_output(x);

            cv::Point max_loc;
            double max;
            cv::minMaxLoc(y, nullptr, &max, nullptr, &max_loc);
            
            if (max_loc.y != label) error++;
        }

        error = (error / n_test) * 100;
        return error;
    }

    double error_train(DigitsDataset & data)
    {
        double error = 0;

        int n_train = data.get_n_train();

        cv::Mat x, y;
        int label;

        data.reset();
        for (int i = 0; i < n_train; i++)
        {
            data.get_next_train_col(x, label);
            y = get_output(x);

            cv::Point max_loc;
            double max;
            cv::minMaxLoc(y, nullptr, &max, nullptr, &max_loc);

            if (max_loc.y != label) error++;
        }

        error = (error / n_train) * 100;
        return error;
    }

    double cross_entropy(cv::Mat y, cv::Mat u)
    {
        double sum = 0;
        for (int i = 0; i < M; i++)
        {
            sum += y.at<double>(i, 0) * log(u.at<double>(i, 0));
        }
        return -sum;
    }

private:
    void train_iteration(cv::Mat x, cv::Mat y, int label, double alpha)
    {
        cv::Mat v(K, 1, CV_64FC1);
        cv::Mat u(M, 1, CV_64FC1);

        cv::Mat grad1(w1.rows, w1.cols, CV_64FC1);
        cv::Mat grad2(w2.rows, w2.cols, CV_64FC1);

        cv::Mat mult = w1 * x;
        v = logistic(mult); //hidden layer output
        cv::Mat der1 = logistic_der(mult);

        mult = w2 * v;
        u = softmax(mult); // net output
        cv::Mat der2 = softmax_der(mult);

        for (int j = 0; j < grad2.rows; j++)
            for (int s = 0; s < grad2.cols; s++)
            {
                grad2.at<double>(j, s) = (u.at<double>(j, 0) - y.at<double>(j, 0)) * v.at<double>(s, 0);
            }

        for (int s = 0; s < grad1.rows; s++)
            for (int i = 0; i < grad1.cols; i++)
            {
                grad1.at<double>(s, i) = 0;
                for (int r = 0; r < M; r++)
                {
                    grad1.at<double>(s, i) += (u.at<double>(r, 0) - y.at<double>(r, 0)) * w2.at<double>(r, s) * der1.at<double>(s, 0) * x.at<double>(i, 0);
                }
            }

        w1 -= alpha * grad1;
        w2 -= alpha * grad2;
    }

    cv::Mat softmax(cv::Mat x)
    {
        int tmp = 0;

        cv::Mat res(x.rows, 1, CV_64FC1);

        double min, max;
        cv::minMaxLoc(x, &min, &max, nullptr, nullptr);

        double sum = 0;
        for (int i = 0; i < x.rows; i++)
        {
            sum += exp(x.at<double>(i, 0) - max);
        }

        for (int i = 0; i < x.rows; i++)
        {
            res.at<double>(i, 0) = exp(x.at<double>(i, 0) - max) / sum;
        }

        return res;
    }

    cv::Mat softmax_der(cv::Mat x)
    {
        cv::Mat res = softmax(x);
        for (int i = 0; i < x.rows; i++)
        {
            res.at<double>(i, 0) = res.at<double>(i, 0) * (1 - res.at<double>(i, 0));
        }

        return res;
    }

    cv::Mat logistic(cv::Mat x)
    {
        cv::Mat res(x.rows, 1, CV_64FC1);

        for (int i = 0; i < x.rows; i++)
        {
            res.at<double>(i, 0) = 1 / (1 + exp(-x.at<double>(i, 0)));
        }

        return res;
    }

    cv::Mat logistic_der(cv::Mat x)
    {
        cv::Mat res = logistic(x);

        for (int i = 0; i < x.rows; i++)
        {
            res.at<double>(i, 0) = res.at<double>(i, 0) * (1 - res.at<double>(i, 0));
        }

        return res;
    }

    cv::Mat create_y(int label)
    {
        cv::Mat res(M, 1, CV_64FC1, cv::Scalar(0));
        res.at<double>(label, 0) = 1.0;
        return res;
    }

    cv::Mat w1, w2;
    int N, K, M;
};
