#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


int main() {


	VideoCapture cap("kalmanfilter2.mp4"); //Windowsの場合　パス中の¥は重ねて¥¥とする
	/*if (!cap.isOpened){
		cout << "cap" << endl;
		return -1;
	}*/
	int max_frame = cap.get(CV_CAP_PROP_FRAME_COUNT); //フレーム数
	for (int i = 0; i<max_frame;i++) {
		Mat img;
		cap >> img; //1フレーム分取り出してimgに保持させる
		imshow("Video", img);
		waitKey(1); // 表示のために1ms待つ
	}
	
	return 0;
}