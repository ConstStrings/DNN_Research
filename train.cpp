#include "train.h"

uchar temp[1000];

TrainSet::TrainSet(std::string Path_label, std::string Path_img)
{
	fopen_s(&LabelSet, Path_label.c_str(), "rb");
	fopen_s(&ImaegSet, Path_img.c_str(), "rb");

	fread(&(this->magic_number), 4, 1, LabelSet);
	fread(&(this->labels_num), 4, 1, LabelSet);
	std::cout << "Magic Number:" << this->magic_number << "    Labels Number:" << this->labels_num << std::endl;
	fread(temp, 16, 1, ImaegSet);
}

TrainSet::~TrainSet()
{
	fclose(this->ImaegSet);
	fclose(this->LabelSet);
}

Mat to_mat(uchar* buffer)
{
	cv::Mat img;
	img.create(cv::Size(1, 784), CV_8UC1);
	for (int i = 0; i < 784; i++)
	{
		img.at<uchar>(i,0) = *(buffer + i);
	}
	return img;
}

void TrainSet::LoadBatch(batch& b)
{

	for (int i = 0; i < batch_size; i++)
	{
		uchar data;
		fread(&(data), 1, 1, LabelSet);
		b.label.push_back(data);
		//std::cout << (int)data << std::endl;
	}
	for (int i = 0; i < batch_size; i++)
	{
		uchar buffer[784];
		fread(buffer, 1, 784, ImaegSet);
		b.img.push_back(to_mat(buffer));
	}
}