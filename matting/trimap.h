#pragma once
#include <eigen/Eigen/Dense>
#include <opencv2/core.hpp> 

/**
 * Extractas pixels from an image, given a trimap. Pixels is extracted
 * is correcponding trimap value equals val
 * @param img opencv Mat with image - 3 channels
 * @param trimap opencv Mat with trimap - grayscale
 * @param val trimap filter value
 * @return matrix Nx3
 */
std::vector<std::vector<float>> extract_pixels(cv::Mat const& img, cv::Mat const& trimap, float val);

/**
 * Converts 3ch pixels container in to opencv Mat
 * @param pixels vector with pixels from extract_pixels func
 * @return opencv Mat
 */
cv::Mat to_mat(std::vector<std::vector<float>> pixels);

/**
 * Finds mean and inverse covariance matrix
 * for a set of 3 channel pixels
 * @param pixels vector with pixels from extract_pixels func
 * @return tuple with mu and inverse covariance in Eigen types 
 */
std::tuple<Eigen::VectorXf, Eigen::MatrixXf> fit_gauss(std::vector<std::vector<float>> pixels);