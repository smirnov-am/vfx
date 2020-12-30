#pragma once
#include <opencv2/core.hpp> 
#include <eigen/Eigen/Dense>

class BayesMatting {
        Eigen::MatrixXf cov_f_inv, cov_b_inv;

        Eigen::Vector3f mu_f, mu_b;
        float epsilon; // alpha convergance error
        float sigma; // deviation from matting equation
        cv::Mat trimap; // holds initial manual trimap

        /**
         * Populates A in Ax=Y equation
         * @param alpha value for a pixel
         * @param sigma deviation from matting equation
         * @return Eigen 6x6 matrix
         */ 
        Eigen::MatrixXf A(float const& alpha);

        /**
         * Populates Y in Ax=Y equation
         * @param alpha value for a pixel
         * @param sigma deviation from matting equation
         * @param I pixel with 3 [0, 1] normalized channels
         * @return Eigen 6x1 vector
         */
        Eigen::VectorXf Y(float const& alpha, Eigen::Vector3f const& I);

        /**
         * Calculates alpha for a pixel. 
         * Converges when alpha stabilizes 
         * @param alpha_guess - initial guess for alpha from trimap or previous frame
         * @param sigma deviation from matting equation
         * @param pixel with 3 [0, 1] normalized channels
         * @return calulated alpha for this pixel
         */
        float calc_alpha_pixel(float const& alpha_guess, cv::Vec3f const& pixel);

    public:
        BayesMatting(float epsilon, float sigma, cv::Mat trimap): epsilon(epsilon), sigma(sigma), trimap(trimap) {};
        
        /**
         * Call this method before generating alpha maps.
         * It analyzes a keyframe and builds a prior
         * @param keyframe opencv mat with keyframe
         */
        void model(cv::Mat keyframe);

        /**
         * Calcualtes alpha map for a video frame. 
         * Feed consecutive frames to it 
         * @param frame opencv mat with a frame
         */
        cv::Mat alpha_map(cv::Mat frame);
};