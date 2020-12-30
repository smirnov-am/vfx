#include "trimap.h"

using namespace std;
using namespace cv;
using namespace Eigen;


vector<vector<float>> extract_pixels(Mat const& img, Mat const& trimap, float val) 
{   
    vector<vector<float>> pixels{{}, {}, {}};

    for(int i=0; i<img.rows; i++) 
    {
        for(int j=0; j<img.cols; j++) 
        {   

            if (trimap.at<float>(i, j) == val)
            {
                for (int c=0;c<3;c++) {
                    pixels[c].push_back(static_cast<float>(img.at<Vec3f>(i,j)[c]));
                }
            }        
        }        
    }
    return pixels;
}


Mat to_mat(vector<vector<float>> pixels)
{
    Mat result = Mat::zeros(pixels[0].size(), 3, CV_32F);
    for(int i=0; i<result.rows; i++)
        for(int j=0; j<result.cols; j++)
          result.at<float>(i, j) = pixels.at(j).at(i);

    return result;
}


tuple<VectorXf, MatrixXf> fit_gauss(vector<vector<float>> pixels)
{   

    int n_pixels = pixels[0].size();
    
    MatrixXf res_pixels(n_pixels, 3);
    
    for (int c=0;c<3;c++) 
    {
        res_pixels.col(c) = Map<VectorXf>(pixels[c].data(), n_pixels);
    }
    auto mu = res_pixels.colwise().mean();
    MatrixXf centered = res_pixels.rowwise() - mu;
    MatrixXf cov = (centered.adjoint() * centered) / double(res_pixels.rows() - 1);

    return make_tuple(mu, cov.inverse());
}