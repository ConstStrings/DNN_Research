#pragma once

#include <iostream>
#include <fstream> 
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <vector>

using namespace cv;

#define batch_size 100

class batch {
public:
	std::vector<cv::Mat> img;
	std::vector<uchar> label;
};

class TrainSet {
public:
	TrainSet(std::string Path_label, std::string Path_img);
	void LoadBatch(batch &b);
	~TrainSet();

	FILE* LabelSet;
	FILE* ImaegSet;
	int magic_number;
	int labels_num;
};

