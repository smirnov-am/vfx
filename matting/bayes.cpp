#include <iostream> 
#include "trimap.h"
#include "matting.h"

using namespace std;
using namespace cv;
using namespace Eigen;

MatrixXf BayesMatting::A(float const& alpha) 
{
    MatrixXf res(6, 6);
    res.block(0, 0, 3, 3) = cov_f_inv + (alpha*alpha/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    res.block(0, 3, 3, 3) = (alpha*(1-alpha)/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    res.block(3, 0, 3, 3) = (alpha*(1-alpha)/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    res.block(3, 3, 3, 3) = cov_b_inv + ((1-alpha)*(1-alpha)/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    return res;
}

VectorXf BayesMatting::Y(float const& alpha, Vector3f const& I) 
{
    VectorXf res(6);
    res.head(3) = cov_f_inv*mu_f + (alpha/(sigma*sigma)*I);
    res.tail(3) = cov_b_inv*mu_b + ((1-alpha)/(sigma*sigma)*I);
    return res;
}

float BayesMatting::calc_alpha_pixel(float const& alpha_guess, Vec3f const& pixel) 
{
    float next_alpha = alpha_guess + 2*epsilon;
    float alpha = alpha_guess;

    VectorXf FB(6); 

    Vector3f I(pixel[0], pixel[1], pixel[2]);
    while (fabs(next_alpha - alpha) > epsilon) 
    {
        alpha = next_alpha;
        MatrixXf a = A(alpha);
        VectorXf y = Y(alpha, I);
        FB = a.llt().solve(y);
        
        Vector3f F = FB.head(3);
        Vector3f B = FB.tail(3);
        next_alpha = ( (I - B).dot( (F - B) ) ) / (F - B).dot( (F - B) );
        
        if (next_alpha < 0)
        {
            next_alpha = 0.0;
        }
        else if (next_alpha>1) {
            next_alpha = 1.0;
        }
    }
    return alpha;
};

void BayesMatting::model(Mat keyframe) {
    cout << "Processing trimap" << endl;
    auto tri_foreground = extract_pixels(keyframe, trimap, 1.0);
    auto tri_background = extract_pixels(keyframe, trimap, 0);

    cout << "Found " << tri_foreground[0].size() << " pixels marked as foreground by trimap" << endl;
    cout << "Found " << tri_background[0].size() << " pixels marked as background by trimap" << endl;

    auto gauss_bg = fit_gauss(tri_background);
    auto gauss_fg = fit_gauss(tri_foreground);

    mu_f = get<0>(gauss_fg);
    mu_b = get<0>(gauss_bg);
    cov_f_inv = get<1>(gauss_fg);
    cov_b_inv = get<1>(gauss_bg);
    
    cout << "Foreground mu " << endl << mu_f << endl;
    cout << "Foreground inverted covariance " << endl << cov_f_inv << endl;
    cout << "Background mu " << endl << mu_b << endl;
    cout << "Background inverted covariance " << endl << cov_b_inv << endl;

};

Mat BayesMatting::alpha_map(Mat frame)
{
    Mat alpha_map = Mat::zeros(frame.size(), CV_32F);
    int counter = 0;
    float progress = 0.0;

    for(int i=0; i<frame.rows; i++) 
    {
        for(int j=0; j<frame.cols; j++) 
        {   

            Vec3f pixel = frame.at<Vec3f>(i,j);
            float alpha_guess = trimap.at<float>(i, j);

            alpha_map.at<float>(i, j) = calc_alpha_pixel(alpha_guess, pixel);;
            
            progress = 100.0*counter/(frame.rows*frame.cols);
            if (fmod(progress, 2.0) == 0) 
            {
                cout << "               \r"<< progress  << "%" <<flush;
            }
            counter ++;
            
        }        
    }
    cout << endl;

    // trimap for next frame in video is 
    // a current frame alpha map
    trimap = alpha_map.clone();

    return alpha_map;
}
