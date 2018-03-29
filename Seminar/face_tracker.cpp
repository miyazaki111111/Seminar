#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


int main(int argc, char* argv[])
{
	if (argc < 2) {
		std::cout << "exe [video]" << std::endl;
		return -1;
	}

	// Trackerの生成
	cv::Ptr<cv::Tracker> trackerKCF = cv::Tracker::create("KCF");
	cv::Ptr<cv::Tracker> trackerTLD = cv::Tracker::create("TLD");
	cv::Ptr<cv::Tracker> trackerMEDIANFLOW = cv::Tracker::create("MEDIANFLOW");
	cv::Ptr<cv::Tracker> trackerBOOSTING = cv::Tracker::create("BOOSTING");
	cv::Ptr<cv::Tracker> trackerMIL = cv::Tracker::create("MIL");

	cv::VideoCapture cap(argv[1]);
	if (!cap.isOpened()) {
		std::cout << "ビデオが開けません。" << std::endl;
		return -1;
	}


	cv::Mat frame;
	cap >> frame;

	// 最初のフレームから追跡対象を選択
	cv::Rect2d roi = cv::selectROI("tracker", frame); // マウスで中心を選択してドラッグ
	cv::Rect2d roiTLD = roi;
	cv::Rect2d roiMEDIANFLOW = roi;
	cv::Rect2d roiBOOSTING = roi;
	cv::Rect2d roiMIL = roi;

	cv::Mat target(frame, roi);
	cv::imwrite("target.jpeg", target);
	std::cout << "(x, y, width, height) = (" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << ")" << std::endl;

	if (roi.width == 0 || roi.height == 0)
		return -1;

	// Trackerの初期化
	trackerKCF->init(frame, roi);
	trackerTLD->init(frame, roiTLD);
	trackerMEDIANFLOW->init(frame, roiMEDIANFLOW);
	trackerBOOSTING->init(frame, roiBOOSTING);
	trackerMIL->init(frame, roiMIL);

	// Trackerごとの色
	cv::Scalar colorkcf = cv::Scalar(0, 255, 0);
	cv::Scalar colortld = cv::Scalar(0, 255, 255);
	cv::Scalar colormedianflow = cv::Scalar(0, 0, 255);
	cv::Scalar colorboosting = cv::Scalar(255, 255, 0);
	cv::Scalar colormit = cv::Scalar(255, 0, 255);

	// 動画保存の設定
	double fps = cap.get(CV_CAP_PROP_FPS);
	cv::Size size = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	const int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
	std::string filename = "output.avi";
	cv::VideoWriter writer(filename, fourcc, fps, size);

	while (1) {
		cap >> frame;
		if (frame.empty())
			break;

		// 更新
		trackerKCF->update(frame, roi);
		trackerTLD->update(frame, roiTLD);
		trackerMEDIANFLOW->update(frame, roiMEDIANFLOW);
		trackerBOOSTING->update(frame, roiBOOSTING);
		trackerMIL->update(frame, roiMIL);

		// 矩形で囲む
		cv::rectangle(frame, roi, colorkcf, 1, 1);
		cv::rectangle(frame, roiTLD, colortld, 1, 1);
		cv::rectangle(frame, roiMEDIANFLOW, colormedianflow, 1, 1);
		cv::rectangle(frame, roiBOOSTING, colorboosting, 1, 1);
		cv::rectangle(frame, roiMIL, colormit, 1, 1);

		cv::putText(frame, "- KCF", cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, .5, colorkcf, 1, CV_AA);
		cv::putText(frame, "- TLD", cv::Point(65, 20), cv::FONT_HERSHEY_SIMPLEX, .5, colortld, 1, CV_AA);
		cv::putText(frame, "- MEDIANFLOW", cv::Point(125, 20), cv::FONT_HERSHEY_SIMPLEX, .5, colormedianflow, 1, CV_AA);
		cv::putText(frame, "- BOOSTING", cv::Point(5, 40), cv::FONT_HERSHEY_SIMPLEX, .5, colorboosting, 1, CV_AA);
		cv::putText(frame, "- MIL", cv::Point(115, 40), cv::FONT_HERSHEY_SIMPLEX, .5, colormit, 1, CV_AA);

		cv::imshow("tracker", frame);
		writer << frame;

		int key = cv::waitKey(30);
		if (key == 0x1b) //ESCキー
			break;
	}

	return 0;
}