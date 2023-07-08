#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include "opencv2/opencv.hpp"

typedef unsigned int uint;

const double EPSILON = 1e-8;

struct deriv_params {
    double smooth_sigma_x;
    double smooth_sigma_y;
};

struct harris_corner_params {
    uint neighbourhood_size;
    uint sobel_size;
    // [0.04, 0.06]
    double k;
    double threshold;
};

struct lucas_kanade_params {
    uint neighbourhood_size;
};

struct horn_schunck_params {
    uint iterations;
    double lambda;
};

struct flow_params {
    deriv_params deriv_params;
    lucas_kanade_params lucas_kanade_params;
    horn_schunck_params horn_schunck_params;
    bool sparse;
    harris_corner_params harris_corner_params;
};

void gausssmooth(const cv::Mat& oim, const float sigmax, const float sigmay, cv::Mat& out) {
    cv::GaussianBlur(oim, out, cv::Size(), sigmax, sigmay);
}

void get_deriv(const cv::Mat& oim, const deriv_params& params, cv::Mat& out_dx, cv::Mat& out_dy) {
    cv::Sobel(oim, out_dx, CV_32F, 1, 0, 3);
    cv::Sobel(oim, out_dy, CV_32F, 0, 1, 3);
}

void get_frame_derivs(cv::Mat im1, cv::Mat im2, const deriv_params& params, cv::Mat& out_dx, cv::Mat& out_dy, cv::Mat& out_dt) {
    cv::Mat I1x, I1y, I2x, I2y;

    gausssmooth(im1, params.smooth_sigma_x, params.smooth_sigma_y, im1);
    gausssmooth(im2, params.smooth_sigma_x, params.smooth_sigma_y, im2);

    get_deriv(im1, params, I1x, I1y);
    get_deriv(im2, params, I2x, I2y);

    cv::subtract(im2, im1, out_dt);
    cv::addWeighted(I1x, 0.5, I2x, 0.5, 0.0, out_dx);
    cv::addWeighted(I1y, 0.5, I2y, 0.5, 0.0, out_dy);
}

void harris_corner(const cv::Mat& im1, const cv::Mat& im2, const harris_corner_params& params, cv::Mat& out_response) {
    cv::Mat response1, response2;
    cv::cornerHarris(im1, response1, params.neighbourhood_size, params.sobel_size, params.k, cv::BORDER_REPLICATE);
    cv::cornerHarris(im2, response2, params.neighbourhood_size, params.sobel_size, params.k, cv::BORDER_REPLICATE);
    cv::addWeighted(response1, 0.5, response2, 0.5, 0.0, out_response);
    
    // double min, max;
    // cv::minMaxLoc(out_response, &min, &max);
    // std::cout << "min response: " << min << "max response: " << max << std::endl;
}

void lucaskanade_flow(const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It,  const lucas_kanade_params& params, cv::Mat& out_u, cv::Mat& out_v) {
    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat IxIy = Ix.mul(Iy);
    cv::Mat IxIt = Ix.mul(It);
    cv::Mat IyIt = Iy.mul(It);


    cv::boxFilter(Ix2, Ix2, CV_32F, cv::Size(params.neighbourhood_size, params.neighbourhood_size), cv::Point(-1, -1), false, cv::BORDER_REFLECT);
    cv::boxFilter(IxIy, IxIy, CV_32F, cv::Size(params.neighbourhood_size, params.neighbourhood_size), cv::Point(-1, -1), false, cv::BORDER_REFLECT);
    cv::boxFilter(Iy2, Iy2, CV_32F, cv::Size(params.neighbourhood_size, params.neighbourhood_size), cv::Point(-1, -1), false, cv::BORDER_REFLECT);
    cv::boxFilter(IxIt, IxIt, CV_32F, cv::Size(params.neighbourhood_size, params.neighbourhood_size), cv::Point(-1, -1), false, cv::BORDER_REFLECT);
    cv::boxFilter(IyIt, IyIt, CV_32F, cv::Size(params.neighbourhood_size, params.neighbourhood_size), cv::Point(-1, -1), false, cv::BORDER_REFLECT);

    // D = Ix2 * Iy2 - IxIy^2
    cv::Mat det;
    cv::multiply(Ix2, Iy2, det);
    cv::subtract(det, IxIy.mul(IxIy), det);

    cv::Mat mask = det < EPSILON;
    det = det.setTo(1.0, mask);

    // u = (IxIy * IyIt - Iy2 * IxIt) / D
    out_u = (IxIy.mul(IyIt) - Iy2.mul(IxIt)) / det;
    // v = (IxIy * IxIt - Ix2 * IyIt) / D
    out_v = (IxIt.mul(IxIy) - Ix2.mul(IyIt)) / det;

    out_u = out_u.setTo(0.0, mask);
    out_v = out_v.setTo(0.0, mask);
}

void hornschunck_flow(const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It,  const horn_schunck_params& params, cv::Mat& out_u, cv::Mat& out_v) {
    cv::Mat avg_kernel = (cv::Mat_<double>(3,3) << 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0);

    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);

    // D = Ix2 + Iy2 + lambda
    cv::Mat det = Ix2 + Iy2 + params.lambda;

    cv::Mat u_prev = out_u.clone();
    cv::Mat v_prev = out_v.clone();

    for (uint i = 0; i < params.iterations; ++i) {
        // P = (It + Ix*u + Iy*v) / D
        cv::Mat P = (It + Ix.mul(u_prev) + Iy.mul(v_prev)) / det;

        cv::subtract(u_prev, Ix.mul(P), out_u);
        cv::subtract(v_prev, Iy.mul(P), out_v);

        cv::filter2D(out_u, u_prev, CV_32F, avg_kernel);
        cv::filter2D(out_v, v_prev, CV_32F, avg_kernel);
    }
}

void calculate_flow(cv::Mat im1, cv::Mat im2, const flow_params& params, cv::Mat& out_u, cv::Mat& out_v) {
    // Transform to grayscale float
    cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);
    im1.convertTo(im1, CV_32F, 1.0 / 255.0);
    im2.convertTo(im2, CV_32F, 1.0 / 255.0);
    
    cv::Mat Ix, Iy, It;
    get_frame_derivs(im1, im2, params.deriv_params, Ix, Iy, It);

    lucaskanade_flow(Ix, Iy, It, params.lucas_kanade_params, out_u, out_v);
    if (params.sparse) {
        cv::Mat response;
        harris_corner(im1, im2, params.harris_corner_params, response);
        cv::Mat mask = response < params.harris_corner_params.threshold;
        out_u = out_u.setTo(0.0, mask);
        out_v = out_v.setTo(0.0, mask);
    }
    if (params.horn_schunck_params.iterations > 0) {
        hornschunck_flow(Ix, Iy, It, params.horn_schunck_params, out_u, out_v);
        if (params.sparse) {
            cv::Mat response;
            harris_corner(im1, im2, params.harris_corner_params, response);
            cv::Mat mask = response < params.harris_corner_params.threshold;
            out_u = out_u.setTo(0.0, mask);
            out_v = out_v.setTo(0.0, mask);
        }
    }
}

void show_flow(const cv::Mat& im1, const cv::Mat& u, const cv::Mat& v) {
    cv::Mat im = im1.clone();
    cv::Mat su, sv;
    cv::GaussianBlur(u, su, cv::Size(), 3.0);
    cv::GaussianBlur(v, sv, cv::Size(), 3.0);
    for (uint i = 0; i < su.rows; i += 5) {
        for (uint j = 0; j < su.cols; j += 5) {
            cv::Point2f p1(j, i);
            cv::Point2f p2(j + su.at<float>(i, j), i + sv.at<float>(i, j));
            cv::arrowedLine(im, p1, p2, cv::Scalar(0, 0, 255), 1);
        }
    }
    cv::imshow("Flow", im);
    cv::waitKey(10);
}

#endif