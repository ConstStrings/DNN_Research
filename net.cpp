#include "net.h"

float lr = 1e-2;

void Rand_Init(Mat& matrix)		//Init the weight randly
{
	int m = matrix.rows;
	int n = matrix.cols;
	srand(time(NULL));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float num = (rand() - 0.5 * static_cast<double>(RAND_MAX)) / static_cast<double>(RAND_MAX);
			matrix.at<float>(i, j) = num*5;
		}
	}
}

//Creat two inside layer 784*16 and 16*16
Net::Net()		
{
	layer1.create(cv::Size(784, 16), CV_32FC1);
	Rand_Init(layer1);
	//std::cout << layer1 << std::endl;

	layer2.create(cv::Size(16, 16), CV_32FC1);
	Rand_Init(layer2);

	layer3.create(cv::Size(16, 10), CV_32FC1);
	Rand_Init(layer3);


	bias1.create(cv::Size(1, 16), CV_32FC1);
	Rand_Init(bias1);
	bias2.create(cv::Size(1, 16), CV_32FC1);
	Rand_Init(bias2);
	bias3.create(cv::Size(1, 10), CV_32FC1);
	Rand_Init(bias3);

	out0.create(cv::Size(1, 784), CV_32FC1);
	out0.setTo(Scalar(0));
	out1.create(cv::Size(1, 16), CV_32FC1);
	out1.setTo(Scalar(0));
	out2.create(cv::Size(1, 16), CV_32FC1);
	out2.setTo(Scalar(0));
	out3.create(cv::Size(1, 10), CV_32FC1);
	out3.setTo(Scalar(0));
}

//This activate function is used
float sigmoid(float x) 
{
	return 1 / (1 + std::exp(-x));
}

float activate_det(float x)
{
	return (std::exp(-x)) / (1 + std::exp(-x)) / (1 + std::exp(-x));
}

float ReLU(float a)
{
	return a > 0 ? a : 0;
}

Mat softmax(Mat matrix)
{
	int m = matrix.rows;
	float sum = 0;
	for (int i = 0; i < m; i++)
	{
		sum += matrix.at<float>(i, 0);
	}
	return matrix / sum;
}

//Apply the activate function to the matrix
void normalize(Mat& matrix)
{
	int m = matrix.rows;
	int n = matrix.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix.at<float>(i, j) = sigmoid(matrix.at<float>(i, j));
		}
	}
}

//Use the weight to calculate the next layer
Mat layer_map(Mat input,Mat layer,Mat bias)
{
	int m = layer.rows;
	int n = layer.cols;
	if (input.rows != n || bias.rows != m)
		std::cout << "Error Occur!" << std::endl;

	Mat output = layer * input + bias;

	normalize(output);

	return output;
}

//Not used
Mat to_colvec(Mat matrix)
{
	int m = matrix.rows;
	int n = matrix.cols;
	Mat out(Size(1, 0), matrix.type());
	for (int i = 0; i < n; i++)
	{
		Mat c = matrix.col(i);
		vconcat(out, c, out);
	}
	return out;
}

//Not used
Mat to_img(Mat matrix)
{
	int n = matrix.rows;
	Mat out(Size(28, 28), matrix.type());
	for (int i = 0; i < n; i++)
	{
		out.at<uchar>(i / 28, i % 28) = matrix.at<uchar>(i, 0);
	}
	return out;
}

//To recongnise image
Mat Net::process(Mat img)
{
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	cv::Mat float_image;
	gray.convertTo(float_image, CV_32FC1, 1.0 / 255.0);
	//std::cout << float_image << std::endl;
	Mat vimg = to_colvec(float_image);
	this->out1 = layer_map(vimg, this->layer1, this->bias1);
	this->out2 = layer_map(out1, this->layer2, this->bias2);
	this->out3 = layer_map(out2, this->layer3, this->bias3);
	std::cout << out3 << std::endl;
	return out3;
}

//forward-propagating
Mat Net::forward(Mat img)
{
	cv::Mat float_image;
	img.convertTo(float_image, CV_32FC1, 1.0 / 255.0);
	this->out0 = float_image;
	this->out1 = layer_map(float_image, this->layer1, this->bias1);
	this->out2 = layer_map(this->out1, this->layer2, this->bias2);
	this->out3 = softmax(layer_map(this->out2, this->layer3, this->bias3));
	return this->out3;
}

//Calculate cost
Mat Net::Cost(Mat img,int number)
{
	Mat result = this->forward(img);
	Mat target(Size(1, 10), CV_32FC1, Scalar(0));
	target.at<float>(number, 0) = 1;
	Mat error = target - result;
	//std::cout << error << std::endl;
	return error;
}

void Net::Train(TrainSet& set)
{
	batch b;
	for (int i = 0; i < train_turns; i++)
	{
		std::cout << "========Batch " << i << " Start========" << std::endl;
		set.LoadBatch(b);
		for (int j = 0; j < b.label.size(); j++)
		{
			Mat cost = this->Cost(b.img[j],b.label[j]);	
			//std::cout << to_img(b.img[j]) << b.label[j] << std::endl;
			//std::cout << -cost << std::endl;
			this->backward(-cost);
			this->update_sgd();
		}
	}
}

void Net::backward(Mat error3)
{
	Mat error1(Size(1, 16), CV_32FC1, Scalar(0));    //µÚÒ»Òþ²Ø²ãÎó²î
	Mat error2(Size(1, 16), CV_32FC1, Scalar(0));    //µÚ¶þÒþ²Ø²ãÎó²î

	error2 = this->layer3.t() * error3;

	for (int i = 0; i < 16; i++)
	{
		error2.at<float>(i, 0) *= activate_det(this->out2.at<float>(i, 0));
	}
	
	error1 = this->layer2.t() * error2;

	for (int i = 0; i < 16; i++)
	{
		error1.at<float>(i, 0) *= activate_det(this->out1.at<float>(i, 0));
	}
	this->e1 = error1;
	this->e2 = error2;
	this->e3 = error3;
	/*std::cout << error1 << std::endl;
	std::cout << error2 << std::endl;
	std::cout << error3 << std::endl;*/
}
 
Mat weight_map(Mat weight,Mat error)
{
	Mat s;
	weight.copyTo(s);
	int m = weight.rows;
	int n = weight.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			s.at<float>(i, j) *= lr * error.at<float>(i, 0);
		}
	}
	return s;
}

void Net::update_sgd()
{
	//Update the hidden layer weights
	//std::cout << weight_map(this->layer3, this->e3) << std::endl;
	layer3 -=  weight_map(this->layer3,this->e3);
	layer2 -= weight_map(this->layer2, this->e2);
	//Update input layer weights
	layer1 -= weight_map(this->layer1, this->e1);
	//Update all layer bias
	bias3 -= lr * this->e3;
	bias2 -= lr * this->e2;
	bias1 -= lr * this->e1;
}

void Net::Save()
{
	DataSave save;
	this->layer1.copyTo(save.layer1);
	this->layer2.copyTo(save.layer2);
	this->layer3.copyTo(save.layer3);
	this->bias1.copyTo(save.bias1);
	this->bias2.copyTo(save.bias2);
	this->bias3.copyTo(save.bias3);

	std::ofstream ofs;
	ofs.open("save.dat", std::ios::binary | std::ios::out);
	ofs.write((const char*)save.layer1.data, save.layer1.total() * sizeof(float));
	ofs.write((const char*)save.layer2.data, save.layer2.total() * sizeof(float));
	ofs.write((const char*)save.layer3.data, save.layer3.total() * sizeof(float));
	ofs.write((const char*)save.bias1.data, save.bias1.total() * sizeof(float));
	ofs.write((const char*)save.bias2.data, save.bias2.total() * sizeof(float));
	ofs.write((const char*)save.bias3.data, save.bias3.total() * sizeof(float));
	ofs.close();

	/*std::vector<int> compression_params;
	compression_params.push_back(IMWRITE_EXR_TYPE_FLOAT);

	imwrite("layer1.EXR", this->layer1, compression_params);
	imwrite("layer2.dat", this->layer2,compression_params);
	imwrite("layer3.dat", this->layer3, compression_params);
	imwrite("bias1.dat", this->bias1, compression_params);
	imwrite("bias2.dat", this->bias2, compression_params);
	imwrite("bias3.dat", this->bias3, compression_params);*/

	std::cout << "Save " << (save.layer1.total()+ save.layer2.total()+ save.layer3.total()+ save.bias1.total()+ save.bias2.total()+ save.bias3.total()) * sizeof(float) << " Byte Done!" << std::endl;
}

void Net::ReadWeight()
{
	std::ifstream ifs;

	ifs.open("save.dat", std::ios::in, std::ios::binary);

	if (!ifs.is_open()) {
		std::cout << "Open Fail!" << std::endl;
		return;
	}

	ifs.read((char*)this->layer1.data, this->layer1.total() * sizeof(float));
	ifs.read((char*)this->layer2.data, this->layer2.total() * sizeof(float));
	ifs.read((char*)this->layer3.data, this->layer3.total() * sizeof(float));
	ifs.read((char*)this->bias1.data, this->bias1.total() * sizeof(float));
	ifs.read((char*)this->bias2.data, this->bias2.total() * sizeof(float));
	ifs.read((char*)this->bias3.data, this->bias3.total() * sizeof(float));

	std::cout << "Load " << (layer1.total() + layer2.total() + layer3.total() + bias1.total() + bias2.total() + bias3.total()) * sizeof(float) << " Byte Done!" << std::endl;

	ifs.close();
}

void Net::ShowLayer()
{
	std::cout << "========Layer========" << std::endl;
	std::cout << layer1 << std::endl;
	std::cout << layer2 << std::endl;
	std::cout << layer3 << std::endl;
}

DataSave::DataSave()
{
	layer1.create(cv::Size(784, 16), CV_32FC1);
	layer1.setTo(Scalar(0));
	layer2.create(cv::Size(16, 16), CV_32FC1);
	layer2.setTo(Scalar(0));
	layer3.create(cv::Size(16, 10), CV_32FC1);
	layer3.setTo(Scalar(0));

	bias1.create(cv::Size(1, 16), CV_32FC1);
	bias1.setTo(Scalar(0));
	bias2.create(cv::Size(1, 16), CV_32FC1);
	bias2.setTo(Scalar(0));
	bias3.create(cv::Size(1, 10), CV_32FC1);
	bias3.setTo(Scalar(0));
}
