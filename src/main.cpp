#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <ctime>
#include <iostream>

#include "DigitsDataset.h"
#include "NeuralNet.h"


int main(int argc, char **argv)
{
    if (argc == 1)
    {
        std::cout << "-d  path to dataset" << std::endl;
        std::cout << "-n  size of a hidden layer" << std::endl;
        std::cout << "-i  trainig iterations count" << std::endl;
        std::cout << "-s  trainig speed" << std::endl;
        std::cout << std::endl;
        return -1;
    }

    cv::theRNG().state = time(NULL);

    std::string dataset_path("");
    int hidden_layer_size = 49;
    int iter_count = 2000;
    double speed = 0.1;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-d") == 0)
        {
            dataset_path = argv[i + 1];
        }
        else if (strcmp(argv[i], "-n") == 0)
        {
            hidden_layer_size = std::stoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            iter_count = std::stoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-s") == 0)
        {
            speed = std::stod(argv[i + 1]);
        }
    }

    if (dataset_path.empty())
    {
        std::cout << "specify path to dataset" << std::endl << std::endl;
        return -1;
    }

    DigitsDataset dataset(dataset_path);
    double err_test, err_train;

    /*for (int ls = 49; ls <= 294; ls *= 6)
        for (double s = 0.0125; s <= 0.4; s *= 2)
            for (int i = 500; i <= 32000; i *= 2)
            {
                std::cout << "iterations count: " << i << std::endl;
                std::cout << "hidden layer size: " << ls << std::endl;
                std::cout << "speed: " << s << std::endl;

                NeuralNet net(28 * 28, ls, 10);
                net.train(dataset, i, s);

                err_test = net.error_test(dataset);
                err_train = net.error_train(dataset);

                std::cout << "error train = " << err_train << "%" << std::endl;
                std::cout << "error test = " << err_test << "%" << std::endl << std::endl;
            }*/

    std::cout << "iterations count: " << iter_count << std::endl;
    std::cout << "hidden layer size: " << hidden_layer_size << std::endl;
    std::cout << "speed: " << speed << std::endl;

    NeuralNet net(28 * 28, hidden_layer_size, 10);
    net.train(dataset, iter_count, speed);

    err_test = net.error_test(dataset);
    err_train = net.error_train(dataset);
    
    std::cout << "error train = " << err_train << "%" << std::endl;
    std::cout << "error test = " << err_test << "%" << std::endl << std::endl;


    cvWaitKey();
    return 0;
}