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
	// 1. Инициализация словаря и детектора
	aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
	aruco::ArucoDetector detector(dictionary);
	// 2. Открытие видеопотока
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

		// 3. Обнаружение маркеров
		vector<int> markerIds;
		vector<vector<Point2f>> markerCorners, rejectedCandidates;
		detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);

		if (!markerIds.empty()) {
			// 4. Оценка позы маркеров
			vector<Vec3d> rvecs, tvecs;
			aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension,
				cameraMatrix, distCoeffs, rvecs, tvecs);

			// 5. Отрисовка маркеров и осей
			aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

			for (size_t i = 0; i < rvecs.size(); ++i) {
				drawFrameAxes(frame, cameraMatrix, distCoeffs,
					rvecs[i], tvecs[i],
					arucoSquareDimension * 0.5f);
			}
		}

		// 6. Отображение результата
		imshow("ArUco Detection", frame);

		// 7. Выход по ESC
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
		// Сохраняет матрицу камеры

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

void detectAndDrawCircles(Mat& frame) {
	Mat gray, blur, edges;

	// 1. Предварительная обработка
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	medianBlur(gray, blur, 7);  // Медианный фильтр лучше убирает шумы
	Canny(blur, edges, 50, 150); // Детекция границ

	// 2. Детекция кругов с жесткими параметрами
	vector<Vec3f> circles;
	HoughCircles(blur, circles, HOUGH_GRADIENT,
		1,  // dp
		blur.rows / 8,  // minDist (1/8 высоты изображения)
		100, // param1 (порог для Canny)
		30,  // param2 (чем выше, тем строже)
		20,  // minRadius
		150  // maxRadius
	);

	// 3. Дополнительная фильтрация
	vector<Vec3f> validCircles;
	for (const auto& c : circles) {
		// Проверка округлости через моменты
		Rect roi(c[0] - c[2], c[1] - c[2], 2 * c[2], 2 * c[2]);
		if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > edges.cols || roi.y + roi.height > edges.rows)
			continue;

		Mat mask = Mat::zeros(edges.size(), CV_8U);
		circle(mask, Point(c[0], c[1]), c[2], Scalar(255), -1);

		Mat croppedEdges;
		bitwise_and(edges, edges, croppedEdges, mask);

		vector<vector<Point>> contours;
		findContours(croppedEdges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

		if (contours.size() >= 5) {  // Круг должен иметь достаточно границ
			validCircles.push_back(c);
		}
	}

	// 4. Отрисовка только валидных кругов
	for (const auto& c : validCircles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(frame, center, radius, Scalar(0, 255, 0), 2);
		circle(frame, center, 3, Scalar(0, 0, 255), -1);
	}
}

// Функция для детекции и отрисовки прямоугольников
void detectAndDrawRectangles(Mat& frame) {
	Mat gray, blur, edges;

	// 1. Предварительная обработка
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(5, 5), 1);
	Canny(blur, edges, 50, 150);

	// 2. Поиск контуров
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// 3. Фильтрация и аппроксимация
	vector<vector<Point>> rectangles;
	for (const auto& cnt : contours) {
		// Отсеиваем слишком маленькие контуры
		if (contourArea(cnt) < 1000)
			continue;

		vector<Point> approx;
		// Аппроксимируем контур полигоном
		double epsilon = 0.02 * arcLength(cnt, true);
		approxPolyDP(cnt, approx, epsilon, true);

		// Ищем контуры с 4 вершинами и выпуклые
		if (approx.size() == 4 && isContourConvex(approx)) {
			// Проверяем углы между сторонами
			double maxCosine = 0;
			for (int j = 2; j < 5; j++) {
				Point pt1 = approx[j % 4];
				Point pt2 = approx[j - 2];
				Point pt0 = approx[j - 1];

				double dx1 = pt1.x - pt0.x;
				double dy1 = pt1.y - pt0.y;
				double dx2 = pt2.x - pt0.x;
				double dy2 = pt2.y - pt0.y;

				double angle = (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
				maxCosine = max(maxCosine, abs(angle));
			}

			// Если углы близки к прямым (cos ? 0)
			if (maxCosine < 0.3) {
				rectangles.push_back(approx);
			}
		}
	}

	// 4. Отрисовка результатов
	for (const auto& rect : rectangles) {
		for (int i = 0; i < 4; i++) {
			line(frame, rect[i], rect[(i + 1) % 4], Scalar(0, 255, 0), 2);
		}
		// Отрисовка центра
		Moments m = moments(rect);
		Point center(m.m10 / m.m00, m.m01 / m.m00);
		circle(frame, center, 5, Scalar(0, 0, 255), -1);
	}
}



int main() {

	//Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	//Mat distanceCoefficients;

	//cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
	//loadCamerCalibration("CameraCalibration.txt",cameraMatrix, distanceCoefficients);
	//startWebcamMonitoring(cameraMatrix, distanceCoefficients, 0.0085f);

	VideoCapture cap(0, CAP_DSHOW); // Открываем камеру (0 - дефолтная камера)

	if (!cap.isOpened()) {
		cerr << "Error: Could not open camera" << endl;
		return -1;
	}

	Mat frame;
	namedWindow("rectangle detection", WINDOW_AUTOSIZE);

	while (true) {
		cap >> frame; // Захватываем кадр с камеры

		if (frame.empty()) {
			cerr << "Error: Blank frame grabbed" << endl;
			break;
		}

		detectAndDrawRectangles(frame); // Обрабатываем кадр

		imshow("rectangle detection", frame); // Показываем результат

		// Выход по нажатию ESC
		if (waitKey(10) == 27) {
			break;
		}
	}

	cap.release();
	destroyAllWindows();
	
	return 0;
}