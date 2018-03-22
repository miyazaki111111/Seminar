#include "stdafx.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "cap" << endl;
		return -1;
	}
	Mat prev;
	Mat next;
	Mat prev_original;
	
	int count = 0;//画像の保存用
	while (1) {
		Mat src_img;
		cap >> src_img;
		Mat frame = src_img.clone();
		cv::namedWindow("result", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		cv::imshow("result", src_img);
		int typeKey = waitKey(1);
		if (typeKey == 'q')break;
		if (typeKey == 's') {
			if (count == 0) {
				prev_original = frame.clone();
				cv::cvtColor(frame, prev, CV_BGR2GRAY);
			}
			else {
				cv::cvtColor(frame, next, CV_BGR2GRAY);
				break;
			}
			count++;
			
		}
	}
	cv::destroyAllWindows();
	
	std::vector<cv::Point2f> prev_pts;
	std::vector<cv::Point2f> next_pts;

	cv::Size flowSize(30, 30);
	cv::Point2f center = cv::Point(prev.cols / 2., prev.rows / 2.);
	for (int i = 0; i<flowSize.width; ++i) {
		for (int j = 0; j<flowSize.width; ++j) {
			cv::Point2f p(i*float(prev.cols) / (flowSize.width - 1),
				j*float(prev.rows) / (flowSize.height - 1));
			prev_pts.push_back((p - center)*0.9f + center);
		}
	}
	// Lucas-Kanadeメソッド＋画像ピラミッドに基づくオプティカルフロー
	// parameters=default
	cv::Mat status, error;
	cv::calcOpticalFlowPyrLK(prev, next, prev_pts, next_pts, status, error);

	// オプティカルフローの表示
	std::vector<cv::Point2f>::const_iterator p = prev_pts.begin();
	std::vector<cv::Point2f>::const_iterator n = next_pts.begin();
	for (; n != next_pts.end(); ++n, ++p) {
		cv::line(prev_original, *p, *n, cv::Scalar(150, 0, 0), 2);
	}

	cv::namedWindow("optical flow", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::imshow("optical flow", prev_original);
	cv::waitKey(0);
	return 0;
}




	


