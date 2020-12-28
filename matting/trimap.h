#pragma once
#include <eigen/Eigen/Dense>
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp>

/**
 * Extractas pixels from an image, given a trimap. Pixels is extracted
 * is correcponding trimap value equals val
 * @param img opencv Mat with image - 3 channels
 * @param trimap opencv Mat with trimap - grayscale
 * @param val trimap filter value
 * @return matrix Nx3
 */
std::vector<std::vector<float>> extract_pixels(cv::Mat const& img, cv::Mat const& trimap, int val);


cv::Mat to_mat(std::vector<std::vector<float>> pixels);
