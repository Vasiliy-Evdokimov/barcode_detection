/*
 * template.cpp
 *
 *  Created on: 26 дек. 2023 г.
 *      Author: vevdokimov
 */

#include "template.hpp"

using namespace std;
using namespace cv;

std::vector<Template> templates;

const string templates_config_folder = "config/";
const string templates_config_filename = "templates_detection.cfg";
const string templates_config_filepath = templates_config_folder + templates_config_filename;

void templates_load_config()
{
	cout << "templates_config_filepath = " << templates_config_filepath << endl;
	templates.clear();
	//
	std::ifstream file(templates_config_filepath);
	if (!file) {
		cout << "templates_load_config() file open error!" << endl;
	} else {
		string s;
		int i;
		double n, x, y, w, h;
		while (getline(file, s))
		{
			Template new_template;
			std::istringstream is(s);
			i = 0;
			while (is >> n)
			{
				switch (i++)
				{
					case 0:
						new_template.match = n;
						break;
					case 1:
						x = n;
						break;
					case 2:
						y = n;
						break;
					case 3:
						w = n;
						break;
					case 4:
						h = n;
						//
						new_template.roi = Rect(x, y, w, h);
						break;
				}
			}
			templates.push_back(new_template);
		}
		//
		file.close();
		cout << "templates_load_config() successfully! templates.size = " << templates.size() << endl;
		cout << "templates_matches: " << endl;
		for (size_t j = 0; j < templates.size(); j++)
			cout << templates[j].match << endl;
	}
	//
	for (size_t i = 0; i < templates.size(); i++) {
		string filepath = templates_config_folder + to_string(i) + ".jpg";
		templates[i].image = imread(filepath);
	}
}

void templates_save_config()
{
	cout << "templates_config_filepath = " << templates_config_filepath << endl;
	//
	std::ofstream out;
	out.open(templates_config_filepath);
	if (!out.is_open()) {
		cout << "templates_save_config() file open error!" << endl;
	} else {
		//
		for (size_t i = 0; i < templates.size(); i++)
		{
			Template tmpl = templates[i];
			out << tmpl.match << " "
				<< tmpl.roi.x << " " << tmpl.roi.y << " "
				<< tmpl.roi.width << " " << tmpl.roi.height
				<< endl;
		}
		//
		out.close();
		cout << "templates_save_config() successfully!" << endl;
	}
	//
	for (size_t i = 0; i < templates.size(); i++) {
		string filepath = templates_config_folder + to_string(i) + ".jpg";
		imwrite(filepath, templates[i].image);
	}
}

void templates_detect(cv::Mat& srcImg, std::vector<DetectionResult>& detection_results)
{
	detection_results.clear();
	//
	Mat img, templ, mask, result;
	//
	int match_method = 5;
//	0 TM_SQDIFF; M
//	1 TM_SQDIFF_NORMED;
//	2 TM_CCORR;
//	3 TM_CCORR_NORMED; M
//	4 TM_CCOEFF;
//	5 TM_CCOEFF_NORMED;
	//
	bool use_mask = false;

	for (size_t i = 0; i < templates.size(); i++)
	{
		img = srcImg(templates[i].roi);
		templ = templates[i].image;
		//
		Mat img_display;
		img.copyTo( img_display );
		int result_cols = img.cols - templ.cols + 1;
		int result_rows = img.rows - templ.rows + 1;
		result.create( result_rows, result_cols, CV_32FC1 );
		bool method_accepts_mask = (TM_SQDIFF == match_method || match_method == TM_CCORR_NORMED);
		if (use_mask && method_accepts_mask)
		{
			matchTemplate( img, templ, result, match_method, mask);
		}
		else
		{
			matchTemplate( img, templ, result, match_method);
		}
		//normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc; double matchVal;
		minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
		if ( match_method  == TM_SQDIFF || match_method == TM_SQDIFF_NORMED )
		{
			matchLoc = minLoc;
			matchVal = minVal;
		}
		else
		{
			matchLoc = maxLoc;
			matchVal = maxVal;
		}

//		cout << "template_" << i <<
//			" minVal=" << minVal <<
//			" maxVal=" << maxVal <<
//			" matchVal=" << matchVal
//		<< endl;

		if (matchVal >= templates[i].match)
		{
			DetectionResult new_result;
			new_result.template_id = i;
			new_result.found_rect = cv::Rect(
				matchLoc + templates[i].roi.tl(),
				Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ) + templates[i].roi.tl()
			);
			new_result.match = matchVal;
			detection_results.push_back(new_result);
			//
			cout << "template_" << to_string(i) << " found! " << matchVal << endl;
		}
	}

}
