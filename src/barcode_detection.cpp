#include <thread>

#include <iostream>
#include <string>

#include <fstream>
#include <vector>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

#include "template.hpp"

using namespace std;
using namespace cv;

#define CLR_BLACK	(cv::Scalar(0x00, 0x00, 0x00))
#define CLR_RED		(cv::Scalar(0x00, 0x00, 0xFF))
#define CLR_BLUE	(cv::Scalar(0xFF, 0x00, 0x00))
#define CLR_GREEN	(cv::Scalar(0x00, 0xFF, 0x00))
#define CLR_YELLOW	(cv::Scalar(0x00, 0xFF, 0xFF))
#define CLR_MAGENTA	(cv::Scalar(0xFF, 0x00, 0xFF))
#define CLR_CYAN	(cv::Scalar(0xFF, 0xFF, 0x00))
#define CLR_WHITE	(cv::Scalar(0xFF, 0xFF, 0xFF))

cv::Scalar templates_clr[] { CLR_YELLOW, CLR_MAGENTA, CLR_CYAN, CLR_BLUE, CLR_GREEN };

string CAMERA_ADDRESS = "rtsp://admin:1234qwer@192.168.1.63:554/streaming/channels/2";
string calibration_file_path = "/home/vevdokimov/eclipse-workspace/line_detection/config/calibration.xml";

cv::Mat cameraMatrix;
cv::Mat distCoeffs;

const std::string IMAGES_FOLDER_NAME = "imgs";
#define COLLECT_INSTRUCTION "press SPACE to capture the image, ESC to break"
#define COLLECT_IMAGES_WND "Collect images - " COLLECT_INSTRUCTION

mutex input_buf_mtx;

int const IMAGE_SIDE = 120;

int collect_images()
{

	struct stat sb;
	if (stat(IMAGES_FOLDER_NAME.c_str(), &sb) == -1)
	{

#ifdef _WIN32
        std::wstring stemp = std::wstring(IMAGES_FOLDER_NAME.begin(), IMAGES_FOLDER_NAME.end());
        CreateDirectory(stemp.c_str(), NULL);
#elif __linux__
        mkdir(IMAGES_FOLDER_NAME.c_str(), 0700);
#endif

	}

    std::cout << "Images collecting started - " << COLLECT_INSTRUCTION << std::endl;

    VideoCapture capture = VideoCapture(CAMERA_ADDRESS);

    char buf[255];

    cv::Mat frame, undistorted, roi_img, save_img;

    capture >> frame;

    Rect roi_rect(
		frame.cols / 2 - IMAGE_SIDE / 2,
		frame.rows / 2 - IMAGE_SIDE / 2,
		IMAGE_SIDE,
		IMAGE_SIDE
	);

    int count = 0;
    int bar = 0;

    while (true)
    {
        capture >> frame;

        if (capture.read(frame)) {

			undistort(frame, undistorted, cameraMatrix, distCoeffs);

		}

	#ifdef _WIN32
        sprintf_s
	#elif __linux__
		sprintf
	#endif
		(buf, "%s/bar%d.%d.jpg", IMAGES_FOLDER_NAME.c_str(), bar, count);

        roi_img = undistorted.clone();
        rectangle(roi_img, roi_rect.tl(), roi_rect.br(), CLR_GREEN, 2);
        string str(buf);
        putText(roi_img, str, Point(10, 15), 1, 1, CLR_GREEN, 1);

		imshow(COLLECT_IMAGES_WND, roi_img);


        int key = waitKey(1);

        if (key != -1)
        	std::cout << "key = " << key << std::endl;

        if (key == 27)
            break;

        if (key == 98)	//	b
        	bar++;

        if (key == 120)	//	x
        	bar = 0;

        if (key == ' ')
        {
            count++;

            save_img = undistorted(roi_rect);

            std::cout << buf << std::endl;
            imwrite(buf, save_img);
            //
            // blink
            bitwise_not(roi_img, roi_img);
            imshow(COLLECT_IMAGES_WND, roi_img);
            waitKey(50);
        }
    }

    cv::destroyWindow(COLLECT_IMAGES_WND);

    std::cout << "Images collecting finished! " << count << " images collected." << std::endl;

    return 0;
}

float input_buf[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
float input_buf2[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
ei_impulse_result_t result2;

bool threads_stop_fl = false;

// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float* out_ptr)
{
	for (size_t i = 0; i < length; i++) {
		out_ptr[i] = (input_buf + offset)[i];
    }
    //
    return EIDSP_OK;
}

int edge_impulse(bool show_results = false)
{
	signal_t signal;            // Wrapper for raw input buffer
	ei_impulse_result_t result; // Used to store inference output
	EI_IMPULSE_ERROR res;       // Return code from inference

	// Calculate the length of the buffer
	size_t buf_len = sizeof(input_buf) / sizeof(input_buf[0]);

	// Make sure that the length of the buffer matches expected input length
	if (buf_len != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
		printf("ERROR: The size of the input buffer is not correct.\n");
		printf("Expected %d items, but got %d\r\n",
				EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
				(int)buf_len);
		return 1;
	}

	// Assign callback function to fill buffer used for preprocessing/inference
	signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
	signal.get_data = &get_signal_data;

	// Perform DSP pre-processing and inference
//	input_buf_mtx.lock();
	res = run_classifier(&signal, &result, false);
//	input_buf_mtx.unlock();

	memcpy(&result2, &result, sizeof(result));

	// Print return code and how long it took to perform inference
	printf("run_classifier returned: %d\n", res);
	printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\n",
			result.timing.dsp,
			result.timing.classification,
			result.timing.anomaly);

	if (show_results)
	{
		// Print the prediction results (object detection)
		printf("Object detection bounding boxes:\r\n");
		for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++) {
			ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
			if (bb.value == 0)
				continue;
			//
			printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
					bb.label,
					bb.value,
					bb.x,
					bb.y,
					bb.width,
					bb.height);
		}
	}

	return 0;
}

void read_file(string filename)
{
	std::ifstream file(filename); // Имя вашего текстового файла
	std::string line;
//	std::vector<int> array;
	int i = 0, j;
	if (file.is_open()) {
		while (std::getline(file, line)) {
			// Используем std::stringstream для разделения строки по запятым
			std::stringstream ss(line);
			std::string token;

			while (std::getline(ss, token, ',')) {
				j = std::stoi(token, nullptr, 16); // Преобразуем строку в число и добавляем в массив
//				array.push_back(j);
				input_buf[i++] = j;
			}
		}
		file.close();

		std::cout << "EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE = " <<	EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE << std::endl;
//		std::cout << "array.size() = " << array.size() << std::endl;
		std::cout << "i = " << i << std::endl;

	}
	else {
		std::cout << "Error opening error file!" << std::endl;
	}

	edge_impulse(true);

	cout << "read_file(" << filename << ") finished!" << endl;

}

void edge_impulse_func()
{
	while (!threads_stop_fl) {

		edge_impulse();

	}
}

void detect_barcode_ei()
{
	cv::VideoCapture capture;
	cv::Mat frame, undistorted, roi_img;

	const string WND_NAME = "Detect barcode";

	capture.open(CAMERA_ADDRESS);

	for (int i = 0; i < 20; i++)
		capture >> frame;

    Rect roi_rect(
		frame.cols / 2 - IMAGE_SIDE / 2,
		frame.rows / 2 - IMAGE_SIDE / 2,
		IMAGE_SIDE,
		IMAGE_SIDE
	);

    threads_stop_fl = false;

    thread edge_impulse_thread(edge_impulse_func);

    while (!threads_stop_fl) {

		if (capture.read(frame)) {

			undistort(frame, undistorted, cameraMatrix, distCoeffs);

		}

		roi_img = undistorted(roi_rect);
		int i = 0;
		//
//		input_buf_mtx.lock();
		for (int r = 0; r < roi_img.rows; r++)
			for (int c = 0; c < roi_img.cols; c++)
			{
				cv::Vec3b pixel = roi_img.at<cv::Vec3b>(r, c);
				input_buf2[i++] = (pixel[2] << 16) + (pixel[1] << 8) + pixel[0];
			}
//		input_buf_mtx.unlock();
		memcpy(input_buf, input_buf2, sizeof(input_buf2));

		rectangle(undistorted, roi_rect.tl(), roi_rect.br(), CLR_GREEN, 2);

		//

		// Print the prediction results (object detection)
		if (result2.bounding_boxes_count > 0)
		{
			//printf("Object detection bounding boxes:\r\n");
			for (uint32_t i = 0; i < result2.bounding_boxes_count; i++) {
				ei_impulse_result_bounding_box_t bb = result2.bounding_boxes[i];
				if (bb.value == 0) {
					continue;
				}
				printf("%s (%f) [ x: %u, y: %u, width: %u, height: %u ]\n",
						bb.label,
						bb.value,
						bb.x,
						bb.y,
						bb.width,
						bb.height);
				//
				string str(bb.label);
				Scalar clr = (str == "bar1") ? CLR_BLUE : CLR_MAGENTA;
				//
				Point pt(roi_rect.tl().x + bb.x, roi_rect.tl().y + bb.y);
				circle(undistorted, pt, 5, clr, -1, cv::LINE_AA);
				putText(undistorted, str, pt + Point(10, 10),
					cv::FONT_HERSHEY_DUPLEX, 1, clr, 1);

			}
		}
		//

		imshow(WND_NAME, undistorted);

		int key = cv::waitKey(1);

		if (key == 27)
			threads_stop_fl = true;

    }

    capture.release();

    destroyWindow(WND_NAME);

}

void detect_barcode_lines()
{
	cv::VideoCapture capture;
	cv::Mat frame, undistorted;

	capture.open(CAMERA_ADDRESS);

	for (int i = 0; i < 20; i++)
		capture >> frame;

	while (1) {

		if (capture.read(frame)) {

			undistort(frame, undistorted, cameraMatrix, distCoeffs);

			//	cv::imshow("frame", frame);
			//	cv::imshow("undistorted", undistorted);

		}

		const int CANNY_LOW = 10;	//	50;		//	250;	//  нижняя граница распознавания контуров
		const int CANNY_HIGH = 150; //	150; 	//	350;	//  верхняя     -//-
		const int HOUGH_LEVEL = 150; // 	150 	//	30;		//  уровень распознавания прямых линий на контурах

		cv::Point cnt(undistorted.cols / 2, undistorted.rows / 2);

		Mat grayscale;
		cvtColor(undistorted, grayscale, COLOR_BGR2GRAY);
		cv::imshow("grayscale", grayscale);
		//
		Mat binary;
		threshold(grayscale, binary, 128, 255, THRESH_BINARY);
		//cv::imshow("binary", binary);
		//
		cv::Mat edges;
		cv::Canny(binary, edges, CANNY_LOW, CANNY_HIGH);
		cv::imshow("edges", edges);
		//
		std::vector<cv::Vec2f> lines;
		cv::HoughLines(edges, lines, 1, CV_PI / 180, HOUGH_LEVEL, 0, 0);

		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0];
			float theta = lines[i][1];

			double a = cos(theta);
			double b = sin(theta);
			double x0 = a * rho;
			double y0 = b * rho;

			Point2f pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
			Point2f pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

			cv::line(undistorted, pt1, pt2, CLR_RED, 1, cv::LINE_AA, 0);
		}

		cv::imshow("lines", undistorted);

		int key = cv::waitKey(1);

		if (key == 27)
			break;
	}

	capture.release();

	destroyAllWindows();

}

cv::Mat undistorted_template;
std::vector<Point> new_template_points;
int templ_func_id = 0;

//#define UNDISTORT
//#define NO_GUI

void onMouse(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
		//
		if (new_template_points.size() == 0)
		{
			new_template_points.push_back(Point(x, y));
			new_template_points.push_back(Point(x, y));
		}
		else if (new_template_points.size() == 2)
		{
			Rect new_template_roi(
				new_template_points[0],
				new_template_points[1]
			);

			Template new_template;
			//
			Mat ROI(undistorted_template, new_template_roi);
			ROI.copyTo(new_template.image);
			new_template.match = 0.5;
			//
			templates.push_back(new_template);

			new_template_points.clear();
		}
	}
	else if (event == cv::EVENT_RBUTTONDOWN)
	{
		std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
	else if (event == cv::EVENT_MBUTTONDOWN)
	{
		std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
		//
		new_template_points.clear();
	}
	else if (event == cv::EVENT_MOUSEMOVE)
	{
		if (new_template_points.size() == 2)
		{
			new_template_points[1].x = x;
			new_template_points[1].y = y;
		}
	}
}

Mat frame_to_show;

const string TEST_WND_NAME = "Templates Test";

void visualizer_func()
{
	pthread_setname_np(pthread_self(), "visualizer thread");

	namedWindow(TEST_WND_NAME);
	setMouseCallback(TEST_WND_NAME, onMouse);

	while (true)
	{
		if (!frame_to_show.empty())
		{
			Mat frame = frame_to_show.clone();
			//
			cv::imshow(TEST_WND_NAME, frame);
		}
		//
		int key = cv::waitKey(1);

		if (key != -1)
			cout << "key = " << key << endl;

		if (key == 102) // f
			templ_func_id++;

		if (key == 120) // x
			templ_func_id = 0;

		if (key == 115) // s
			templates_save_config();

		if (key == 27)
			break;
	}
}

void detect_template_func()
{
	pthread_setname_np(pthread_self(), "detect_template thread");

	cv::VideoCapture capture;
	cv::Mat frame, undistorted_to_show;

	capture.open(CAMERA_ADDRESS);

	for (int i = 0; i < 20; i++)
		capture.grab();

	clock_t tStart;

	double tt, tt_sum, tt_min, tt_max;
	int tt_cnt = 0;

	double targetfps = 25;
	double tt_elapsed, tt_prev = clock();
	int fps_count = 0;

	while (1)
	{
		capture.grab();

		fps_count++;
		tt_elapsed = (double)(clock() - tt_prev) / CLOCKS_PER_SEC;
		if (tt_elapsed < (1. / targetfps))
			continue;

		capture.retrieve(frame);

//		cout << "fps_count = " << fps_count << " tt_elapsed = " << tt_elapsed << endl;

		fps_count = 0;
		tt_prev = clock();

		//if (capture.read(frame))
		{

			tStart = clock();

			try
			{
	#ifdef UNDISTORT
				undistort(frame, undistorted_template, cameraMatrix, distCoeffs);
	#else
				undistorted_template = frame.clone();
	#endif
			}
			catch (...)
			{
				cout << "Undistortion error!" << endl;
				continue;
			}

			undistorted_to_show = undistorted_template.clone();

			std::vector<DetectionResult> results;
			templates_detect(undistorted_to_show, results);

		#ifndef NO_GUI
			Scalar clr;
			for (size_t i = 0; i < templates.size(); i++)
			{
				clr = templates_clr[ i % (sizeof(templates_clr) / sizeof(templates_clr[0])) ];
				rectangle(undistorted_to_show,
					templates[i].roi.tl(),
					templates[i].roi.br(),
					clr, 2, 8, 0);
			}
			//
			for (size_t i = 0; i < results.size(); i++)
			{
				clr = templates_clr[ results[i].template_id % (sizeof(templates_clr) / sizeof(templates_clr[0])) ];
				//
				rectangle(undistorted_to_show,
					results[i].found_rect.tl(),
					results[i].found_rect.br(),
					clr, 2, 8, 0
				);
				putText(undistorted_to_show,
					"templ_" + to_string(results[i].template_id),
					results[i].found_rect.tl() + Point(10, 15),
					cv::FONT_HERSHEY_SIMPLEX, 0.4, clr, 1
				);
				putText(undistorted_to_show,
					to_string(results[i].match),
					results[i].found_rect.tl() + Point(10, 30),
					cv::FONT_HERSHEY_SIMPLEX, 0.4, clr, 1
				);
			}
		#endif

			tt = (double)(clock() - tStart) / CLOCKS_PER_SEC;
			if (!tt_cnt) { tt_sum = 0; tt_max = tt; tt_min = tt; }
			//
			if (tt > tt_max) tt_max = tt;
			if (tt < tt_min) tt_min = tt;
			tt_sum += tt;
			tt_cnt++;

			if (tt_cnt >= 100)
			{
				cout <<
					" min=" << tt_min <<
					" max=" << tt_max <<
					" avg=" << (tt_sum / tt_cnt)
				<< endl	;
				//
				tt_cnt = 0;
			}

		#ifndef NO_GUI
			if (new_template_points.size() == 2)
				rectangle( undistorted_to_show, new_template_points[0], new_template_points[1], CLR_GREEN, 2, 8, 0 );
			//
			frame_to_show = undistorted_to_show.clone();
			//
//			for (size_t i = 0; i < templates.size(); i++)
//				cv::imshow("templ_" + to_string(i), templates[i].image);
		#endif
		}
	}

}

void template_detector()
{
	templates_load_config();
	//
	thread detect_template_thread(detect_template_func);
#ifndef NO_GUI
	std::this_thread::sleep_for(1s);
	thread visualizer_thread(visualizer_func);
#endif
	//
	if (detect_template_thread.joinable()) detect_template_thread.join();
#ifndef NO_GUI
	if (visualizer_thread.joinable()) visualizer_thread.join();
#endif
	//
	while (true) {
		//
		this_thread::sleep_for(100ms);
		//
	}
}

int main()
{
	cout << "OpenCV: " << cv::getBuildInformation() << endl;

#ifdef UNDISTORT
	cv::FileStorage fs(calibration_file_path, cv::FileStorage::READ);
	fs["cameraMatrix"] >> cameraMatrix;
	fs["distCoeffs"] >> distCoeffs;
	fs.release();
#endif

//	read_file("raw_data/bar1_120.txt");
//	read_file("raw_data/bar2_120.txt");
//	return 0;

//	collect_images();
//	return 0;

//	detect_barcode_ei();
//	return 0;

//	detect_barcode_lines();
//	return 0;

	template_detector();

	return 0;
}
