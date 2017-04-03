#include <iostream>
#include <vector>
#include <tuple>
#include "opencv2/opencv.hpp"

using namespace cv;


struct Horizon {
    double m;
    double b;
    double pitch;
    double bank;
    std::vector<double> scores;
    std::vector<std::tuple<double, double>> grid;
};

// std::vector<std::tuple<double, double>> get_grid(double slope_range, double intercept_range) {
//     std::vector<std::tuple<double, double>> answer
// }


Horizon optimize_scores(cv::Mat img, cv::Mat highres, double m, double b, double scaling_factor) {
    std::cout << "Optimize... " << img.size() << highres.size() << std::endl;
    std::vector<std::tuple<double, double>> grid = std::vector<std::tuple<double, double>>();

    Horizon answer;
    return answer;
}



Horizon optimize_global(cv::Mat img, cv::Mat highres, double scaling_factor) {
    return optimize_scores(img, highres, 0.0, 0.0, scaling_factor);
}

Horizon optimize_local(cv::Mat img, cv::Mat highres, double m, double b, double scaling_factor) {
    return optimize_scores(img, highres, m, b, scaling_factor);
}



Horizon optimize_real_time(cv::Mat img, cv::Mat highres, double m, double b, double scaling_factor) {
    Horizon answer;
    if (m == std::numeric_limits<double>::max()) {
        answer = optimize_global(img, highres, scaling_factor);
    }
    else {
        answer = optimize_local(img, highres, m, b, scaling_factor);
    }
    return answer;
}


cv::Mat add_line_to_frame(cv::Mat frame, double m, double b, double pitch, double bank) {
    return frame;
}


int main(int, char**)
{
    double highres_scale = 0.5;
    double scaling_factor = 0.2;


    cv::VideoCapture cap("../../vid/flying_turn.avi");
    if(!cap.isOpened()) {
        std::cout << "error reading video file" << std::endl;
        return -1;
    }

    // int count = 0;
    double m = std::numeric_limits<double>::max(); // TODO figure out a "NULL" initialization method...
    double b = std::numeric_limits<double>::max();
    cv::Mat edges;
    cv::Mat frame;
    cv::Mat img;
    cv::namedWindow("edges",1);
    
    for(;;)
    {
        cap.read(frame); // cap >> frame; // get a new frame from camera
        // std::cout << "frame shape " << frame.rows << " " << frame.cols << std::endl;
        if(frame.empty()) {
            break;
        }
        // cv::cvtColor(frame, edges, cv::COLOR_BGR2GRAY);
        // cv::GaussianBlur(edges, edges, cv::Size(7,7), 1.5, 1.5);
        // cv::Canny(edges, edges, 0, 30, 3);
        // std::cout << frame.size() << std::endl;

        cv::resize(frame, img, cv::Size(), highres_scale, highres_scale);

        Horizon hor = optimize_real_time(img, img, m, b, 1.0);
        cv::Mat prediction;
        prediction = add_line_to_frame(frame, hor.m, hor.b / (highres_scale * scaling_factor), hor.pitch, hor.bank);

        cv::imshow("edges", frame);

        if (cvWaitKey(33) == 27) {
            break;
        }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    std::cout << "exit" << std::endl;
    return 0;
}