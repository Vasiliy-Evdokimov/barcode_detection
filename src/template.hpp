/*
 * template.hpp
 *
 *  Created on: 26 дек. 2023 г.
 *      Author: vevdokimov
 */

#ifndef TEMPLATE_HPP_
#define TEMPLATE_HPP_

#include <iostream>
#include <string>

#include <fstream>
#include <vector>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

struct Template
{
	cv::Mat image;
	double match;
	cv::Rect roi;
};

struct DetectionResult
{
	int template_id;
	cv::Rect found_rect;
	double match;
};

extern std::vector<Template> templates;

void templates_load_config();

void templates_save_config();

void templates_detect(cv::Mat& srcImg, std::vector<DetectionResult>& detection_results);

#endif /* TEMPLATE_HPP_ */
