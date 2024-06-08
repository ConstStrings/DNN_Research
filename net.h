#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <time.h>
#include <Windows.h>
#include<random>
#include "train.h"
#include <vector>

using namespace cv;

#define train_turns 30

class Net {
public:
	Net();
	Mat process(Mat);
	Mat forward(Mat);
	Mat Cost(Mat,int);
	void Train(TrainSet& set);
	void backward(Mat error3);
	void update_sgd();
	void ReadWeight();
	void Save();
	void ShowLayer();

	Mat layer1;
	Mat layer2;
	Mat layer3;
	Mat bias1;
	Mat bias2;
	Mat bias3;
	Mat out0;
	Mat out1;
	Mat out2;
	Mat out3;
	Mat e1;
	Mat e2;
	Mat e3;
};

class DataSave {
public:
	DataSave();

	Mat layer1;
	Mat layer2;
	Mat layer3;
	Mat bias1;
	Mat bias2;
	Mat bias3;
};
