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
	

	while (1) {
		Mat frame;
		cap >> frame;
		double scale = 4.0;
		cv::Mat gray, smallImg(cv::saturate_cast<int>(frame.rows / scale), cv::saturate_cast<int>(frame.cols / scale), CV_8UC1);
		// グレースケール画像に変換
		cv::cvtColor(frame, gray, CV_BGR2GRAY);
		// 処理時間短縮のために画像を縮小
		cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
		cv::equalizeHist(smallImg, smallImg);//グレースケール画像のヒストグラムを均一化する。

		// 分類器の読み込み
		//std::string cascadeName = "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml"; // Haar-like
		std::string cascadeName = "C:/opencv/build/etc/haarcascades/haarcascade_eye.xml"; // Haar-like

																	   //std::string cascadeName = "./lbpcascade_frontalface.xml"; // LBP
		cv::CascadeClassifier cascade;
		if (!cascade.load(cascadeName)) {
			cout << "error" << endl;
			return -1;
		}
		std::vector<cv::Rect> faces;
		/// マルチスケール（顔）探索xo
		// 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
		cascade.detectMultiScale(smallImg, faces,
			1.1, 2,
			CV_HAAR_SCALE_IMAGE,
			cv::Size(30, 30));

		// 結果の描画
		std::vector<cv::Rect>::const_iterator r = faces.begin();
		for (; r != faces.end(); ++r) {
			cv::Point center;
			int radius;
			center.x = cv::saturate_cast<int>((r->x + r->width*0.5)*scale);
			center.y = cv::saturate_cast<int>((r->y + r->height*0.5)*scale);
			radius = cv::saturate_cast<int>((r->width + r->height)*0.25*scale);
			cv::circle(frame, center, radius, cv::Scalar(80, 80, 255), 3, 8, 0);
		}

		cv::namedWindow("result", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		cv::imshow("result", frame);
		int typeKey = waitKey(1);
		if (typeKey == 'q')break;
	}

	return 0;
}

