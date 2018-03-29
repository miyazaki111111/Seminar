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

	//VideoCapture�N���X�ŉ摜�̉𑜓x���擾
	const int w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	const int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	Mat paint_window(cv::Size(w, h), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Point former_face_point = cv::Point(-1, 0);

	cv::Point face_pos;

	CvKalman *kalman = cvCreateKalman(4, 2, 0);//�ʒu�Ƒ��x��4�����A�ϑ��͈ʒu������2����
	cvSetIdentity(kalman->measurement_matrix, cvRealScalar(1.0));
	cvSetIdentity(kalman->process_noise_cov, cvRealScalar(1e-4));
	cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(0.1));
	cvSetIdentity(kalman->error_cov_post, cvRealScalar(1.0));

	// ���������^�����f��
	kalman->DynamMatr[0] = 1.0; kalman->DynamMatr[1] = 0.0; kalman->DynamMatr[2] = 1.0; kalman->DynamMatr[3] = 0.0;
	kalman->DynamMatr[4] = 0.0; kalman->DynamMatr[5] = 1.0; kalman->DynamMatr[6] = 0.0; kalman->DynamMatr[7] = 1.0;
	kalman->DynamMatr[8] = 0.0; kalman->DynamMatr[9] = 0.0; kalman->DynamMatr[10] = 1.0; kalman->DynamMatr[11] = 0.0;
	kalman->DynamMatr[12] = 0.0; kalman->DynamMatr[13] = 0.0; kalman->DynamMatr[14] = 0.0; kalman->DynamMatr[15] = 1.0;


	while (1) {
		Mat frame;
		cap >> frame;
		Mat dest_img = frame.clone();
		double scale = 4.0;
		cv::Mat gray, smallImg(cv::saturate_cast<int>(dest_img.rows / scale), cv::saturate_cast<int>(dest_img.cols / scale), CV_8UC1);


		// �O���[�X�P�[���摜�ɕϊ�
		cv::cvtColor(dest_img, gray, CV_BGR2GRAY);

		// �\���t�F�[�Y
		const CvMat *prediction = cvKalmanPredict(kalman, 0);

		// �\��
		cv::circle(paint_window, cv::Point(prediction->data.fl[0], prediction->data.fl[1]), 2, cv::Scalar(0, 0, 255), 2);

		// �������ԒZ�k�̂��߂ɉ摜���k��
		cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
		cv::equalizeHist(smallImg, smallImg);//�O���[�X�P�[���摜�̃q�X�g�O�������ψꉻ����B

											 // ���ފ�̓ǂݍ���
		std::string cascadeName = "C:/opencv-3.3.1/build/etc/haarcascades/haarcascade_frontalface_alt.xml"; // Haar-like
																											//std::string cascadeName = "C:/opencv/build/etc/haarcascades/haarcascade_eye.xml"; // Haar-like

																											//std::string cascadeName = "./lbpcascade_frontalface.xml"; // LBP
		cv::CascadeClassifier cascade;
		if (!cascade.load(cascadeName)) {
			cout << "error" << endl;
			return -1;
		}
		std::vector<cv::Rect> faces;
		/// �}���`�X�P�[���i��j�T��xo
		// �摜�C�o�͋�`�C�k���X�P�[���C�Œ��`���C�i�t���O�j�C�ŏ���`
		cascade.detectMultiScale(smallImg, faces,
			1.1, 2,
			CV_HAAR_SCALE_IMAGE,
			cv::Size(10, 10));

		// ���ʂ̕`��
		std::vector<cv::Rect>::const_iterator r = faces.begin();
		for (; r != faces.end(); ++r) {
			cv::Point center;
			int radius;
			center.x = cv::saturate_cast<int>((r->x + r->width*0.5)*scale);
			center.y = cv::saturate_cast<int>((r->y + r->height*0.5)*scale);
			radius = cv::saturate_cast<int>((r->width + r->height)*0.25*scale);
			cv::circle(dest_img, center, radius, cv::Scalar(80, 80, 255), 3, 8, 0);
			/*paint_window.at<Vec3b>(center.y, center.x)[0] = 0;
			paint_window.at<Vec3b>(center.y, center.x)[1] = 0;
			paint_window.at<Vec3b>(center.y, center.x)[2] = 0;*/
			//���ډ�f�ɃA�N�Z�X���āA�`��
			//paint_window.at<Vec3b>(center.y, center.x) =Vec3b(255,0,0);

			/*if (former_face_point.x != -1) {
			cv::line(paint_window, former_face_point, center, cv::Scalar(255, 0, 0), 2, 8, 0);
			}*/
			face_pos.x = center.x;
			face_pos.y = center.y;
			cv::circle(paint_window, center, 2, cv::Scalar(255, 0, 0), 2, 8, 0);
			former_face_point = center;
		}
		// �ϑ��l
		float m[] = { face_pos.x, face_pos.y };
		CvMat measurement = cvMat(2, 1, CV_32FC1, m);

		// �C���t�F�[�Y
		const CvMat *correction = cvKalmanCorrect(kalman, &measurement);

		cv::namedWindow("result", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		cv::namedWindow("paint");
		cv::imshow("paint", paint_window);
		cv::imshow("result", dest_img);
		int typeKey = waitKey(1);
		if (typeKey == 'q')break;
	}

	return 0;
}

