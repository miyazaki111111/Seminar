#include "stdafx.h"
#include<iostream>
#include<opencv2/opencv.hpp>


using namespace std;
using namespace cv;

int main() {
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "cap" << endl;
		return -1;
	}


	int count = 0;//‰æ‘œ‚Ì•Û‘¶—p
	while (1) {
		Mat frame;
		cap >> frame;
		Mat src_img = frame.clone();
	

		cv::imshow("result", src_img);
		int typeKey = waitKey(1);
		if (typeKey == 'q')break;
		else if (typeKey == 's') {
			string picturename = "image/num " + to_string(count) + ".png";
			imwrite(picturename, frame);
			count++;
		}

	
	}
	cv::destroyAllWindows();

	return 0;
}

