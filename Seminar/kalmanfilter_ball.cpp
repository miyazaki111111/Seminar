#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// 初期化


int main() {


	VideoCapture cap("kalmanfilter1.mp4"); //Windowsの場合　パス中の¥は重ねて¥¥とする
	cv::Point circle_center;
	int max_frame = cap.get(CV_CAP_PROP_FRAME_COUNT); //フレーム数

	CvKalman *kalman = cvCreateKalman(4, 2, 0);//位置と速度で4次元、観測は位置だから2次元
	cvSetIdentity(kalman->measurement_matrix, cvRealScalar(1.0));
	cvSetIdentity(kalman->process_noise_cov, cvRealScalar(1e-4));
	cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(0.1));
	cvSetIdentity(kalman->error_cov_post, cvRealScalar(1.0));

	// 等速直線運動モデル
	kalman->DynamMatr[0] = 1.0; kalman->DynamMatr[1] = 0.0; kalman->DynamMatr[2] = 1.0; kalman->DynamMatr[3] = 0.0;
	kalman->DynamMatr[4] = 0.0; kalman->DynamMatr[5] = 1.0; kalman->DynamMatr[6] = 0.0; kalman->DynamMatr[7] = 1.0;
	kalman->DynamMatr[8] = 0.0; kalman->DynamMatr[9] = 0.0; kalman->DynamMatr[10] = 1.0; kalman->DynamMatr[11] = 0.0;
	kalman->DynamMatr[12] = 0.0; kalman->DynamMatr[13] = 0.0; kalman->DynamMatr[14] = 0.0; kalman->DynamMatr[15] = 1.0;

	

	for (int i = 0; i<max_frame;i++) {
		Mat img;
		cap >> img; //1フレーム分取り出してimgに保持させる

		cv::Mat dst_img, work_img;
		dst_img = img.clone();

		cv::cvtColor(img, work_img, CV_BGR2GRAY);

		//// Hough変換のための前処理（画像の平滑化を行なわないと誤検出が発生しやすい）
		//cv::GaussianBlur(work_img, work_img, cv::Size(11, 11), 2, 2);

		// 予測フェーズ
		const CvMat *prediction = cvKalmanPredict(kalman,0);

		// 表示
		cv::circle(dst_img, cv::Point(prediction->data.fl[0], prediction->data.fl[1]), 2, cv::Scalar(255, 0, 0),2);

		// Hough変換による円の検出と検出した円の描画
		std::vector<cv::Vec3f> circles;
		cv::HoughCircles(work_img, circles, CV_HOUGH_GRADIENT, 1, 10, 50, 50,0,50);

		std::vector<cv::Vec3f>::iterator it = circles.begin();
		for (; it != circles.end(); ++it) {
			cv::Point center(cv::saturate_cast<int>((*it)[0]), cv::saturate_cast<int>((*it)[1]));
			int radius = cv::saturate_cast<int>((*it)[2]);
			circle_center.x = center.x;circle_center.y = center.y;
			cv::circle(dst_img, center, 2, cv::Scalar(0, 0, 255), 2);
		}
		// 観測値
		float m[] = { circle_center.x, circle_center.y };
		CvMat measurement = cvMat(2, 1, CV_32FC1, m);

		// 修正フェーズ
		const CvMat *correction = cvKalmanCorrect(kalman, &measurement);

		

		
		cv::namedWindow("result", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		imshow("result", dst_img);
		waitKey(1); // 表示のために1ms待つ
		int typeKey = waitKey(1);
		if (typeKey == 'q')break;
	}

	return 0;
}