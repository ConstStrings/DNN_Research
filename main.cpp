#include <iostream>
#include <opencv2/opencv.hpp>
#include <easyx.h>
#include "net.h"
#include "train.h"

using namespace std;
using namespace cv;

batch b;

void doTrain(Net& n1)
{
	TrainSet set1("train/train-labels.idx1-ubyte", "train/train-images.idx3-ubyte");
	n1.Train(set1);
}

void doRecong(Net& n1)
{
	Mat img = imread("demo.png");
	n1.process(img);
}

void showweight(Net& n1)
{
	n1.ShowLayer();
}

void loadsave(Net& n1)
{
	n1.ReadWeight();
	cout << "==============Load Success===============" << endl;
}

int main()
{
	Net n;
	while (1)
	{
		int command = 0;
		cout << "Enter the Command : " << endl;
		cout << "1.Excute Training." << endl;
		cout << "2.Excute Recongize." << endl;
		cout << "3.Show Weight." << endl;
		cout << "4.Load Saved Data." << endl;
		cout << "5.Save Weight." << endl;
		cin >> command;
		switch (command)
		{
		case 1:
			doTrain(n);
			break;
		case 2:
			doRecong(n);
			break;
		case 3:
			showweight(n);
			break;
		case 4:
			loadsave(n);
			break;
		case 5:
			n.Save();
			break;
		default:
			break;
		}
	}
	return 0;
}