#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\opencv.hpp>


#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

const float calibrationSquareDimension = 0.024f;
//const float arucoSquareDimension = 0.01f;  // 1 cm
const Size chessboardDimensions = Size(6, 9);




void createArucoMarkers() {
	
	Mat outputMarker;

	aruco::Dictionary markerDictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);

	for (int i = 0; i < 50; i++) {
		
		aruco::generateImageMarker(markerDictionary, i, 500, outputMarker, 1);
		ostringstream convert;
		string imageName = "4x4marker_";
		convert << imageName << i << ".jpg";
		imwrite(convert.str(), outputMarker);
	}

}

void createKnownboardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners) {
	
	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++ ) 
		{
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, 
	bool showResults = false) {
	
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++) {
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(6,9), pointBuf ,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		
		if (found) {
			allFoundCorners.push_back(pointBuf);
		}

		if (showResults) {
			drawChessboardCorners(*iter, Size(6, 9), pointBuf, found);
			imshow("Loking for Corners", *iter);
			waitKey(0);
		}

	}
}



int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distCoeffs, float arucoSquareDimension) {
	// 1. Èíèöèàëèçàöèÿ ñëîâàðÿ è äåòåêòîðà
	aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
	aruco::ArucoDetector detector(dictionary);
	// 2. Îòêðûòèå âèäåîïîòîêà
	VideoCapture vid(0, CAP_DSHOW);
	if (!vid.isOpened()) {
		cerr << "ERROR: Could not open video capture device" << endl;
		return -1;
	}
	namedWindow("ArUco Detection", WINDOW_AUTOSIZE);
	while (true) {
		Mat frame;
		if (!vid.read(frame)) {
			cerr << "ERROR: Could not read frame" << endl;
			break;
		}

		// 3. Îáíàðóæåíèå ìàðêåðîâ
		vector<int> markerIds;
		vector<vector<Point2f>> markerCorners, rejectedCandidates;
		detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);

		if (!markerIds.empty()) {
			// 4. Îöåíêà ïîçû ìàðêåðîâ
			vector<Vec3d> rvecs, tvecs;
			aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension,
				cameraMatrix, distCoeffs, rvecs, tvecs);

			// 5. Îòðèñîâêà ìàðêåðîâ è îñåé
			aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

			for (size_t i = 0; i < rvecs.size(); ++i) {
				drawFrameAxes(frame, cameraMatrix, distCoeffs,
					rvecs[i], tvecs[i],
					arucoSquareDimension * 0.5f);
			}
		}

		// 6. Îòîáðàæåíèå ðåçóëüòàòà
		imshow("ArUco Detection", frame);

		// 7. Âûõîä ïî ESC
		if (waitKey(30) == 27) break;
	}

	return 1;
}




void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength,
	Mat& cameraMatrix, Mat& distanceCoefficients) {

	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownboardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix,
		distanceCoefficients, rVectors, tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients) {
	ofstream outStream(name);
	if (outStream) {
		// Ñîõðàíÿåò ìàòðèöó êàìåðû

		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				outStream << cameraMatrix.at<double>(r, c) << endl;
			}
		}
		
		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				outStream << distanceCoefficients.at<double>(r, c) << endl;
			}
		}
		outStream.close();
		return true;
	}
	return false;
}

bool loadCamerCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients) {
	
	ifstream inStream(name);
	if (inStream) {
		uint16_t rows;
		uint16_t columns;

		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r, c) = read;
				cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}

		inStream >> rows;
		inStream >> columns;

		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				double read = 0.0f;
				inStream >> read;
				distanceCoefficients.at<double>(r, c) = read;
				cout << distanceCoefficients.at<double>(r, c) << "\n";
			}
		}
		inStream.close();
		return true;
	}

	return false;

}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat distanceCoefficients) {

	Mat frame;
	Mat drawToFrame;

	vector<Mat> savedImages;

	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	VideoCapture vid(0, CAP_DSHOW);

	if (!vid.isOpened()) {
	}

	int framesPerSecond = 30;

	namedWindow("webcam", WINDOW_AUTOSIZE);

	while (true) {
		if (!vid.read(frame))
			break;

		vector<Vec2f> foundPoints;
		bool found = false;

		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		found = findChessboardCorners(gray, chessboardDimensions, foundPoints, CALIB_CB_FAST_CHECK);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);

		if (found)
			imshow("webcam", drawToFrame);
		else
			imshow("webcam", frame);
		char character = waitKey(1000 / framesPerSecond);

		switch (character) {

		case ' ':
			if (found) {
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);

			}

			break;
		case 13:
			if (savedImages.size() > 15) {
				cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension,
					cameraMatrix, distanceCoefficients);
				bool saved = saveCameraCalibration("CameraCalibration.txt", cameraMatrix, distanceCoefficients);

				if (saved) {
					cout << "calibration file saved successfully!" << endl;
				}

				else {
					cout << "failed to save calibration file!" << endl;
				}

			}
			break;
		case 27:
			break;

		}

	}

}



int main() {

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	Mat distanceCoefficients;

	cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
	loadCamerCalibration("CameraCalibration.txt",cameraMatrix, distanceCoefficients);
	startWebcamMonitoring(cameraMatrix, distanceCoefficients, 0.0085f);

	return 0;
}
