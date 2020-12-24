#include <iostream> 
#include <opencv2/videoio.hpp>
#include <cmath>
#include "trimap.h"


using namespace std;
using namespace cv;
using namespace Eigen;

const float alpha_delta = 0.01;

MatrixXf cov_f_inv(3, 3);
MatrixXf cov_b_inv(3, 3);

Vector3f mean_f(3);
Vector3f mean_b(3);


/**
 * Finds mean and inverse covariance matrix
 * for a set of 3 channel pixels
 * @param eigen matrix Nx3 with pixel values [0, 1] normalized
 * @return tuple with mu and inverse covariance in Eigen types 
 */
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

/**
 * Populates A in Ax=Y equation
 * @param alpha value for a pixel
 * @param sigma deviation from matting equation
 * @return Eigen 6x6 matrix
 */ 
MatrixXf A(float const& alpha, float const&  sigma) 
{
    MatrixXf res(6, 6);
    res.block(0, 0, 3, 3) = cov_f_inv + (alpha*alpha/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    res.block(0, 3, 3, 3) = (alpha*(1-alpha)/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    res.block(3, 0, 3, 3) = (alpha*(1-alpha)/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    res.block(3, 3, 3, 3) = cov_b_inv + ((1-alpha)*(1-alpha)/(sigma*sigma))*Matrix<float, 3, 3>::Identity();
    return res;
}

/**
 * Populates Y in Ax=Y equation
 * @param alpha value for a pixel
 * @param sigma deviation from matting equation
 * @param I pixel with 3 [0, 1] normalized channels
 * @return Eigen 6x1 vector
 */
VectorXf Y(float const& alpha, float const& sigma, Vector3f const& I) 
{
    VectorXf res(6);
    res.head(3) = cov_f_inv*mean_f + (alpha/(sigma*sigma)*I);
    res.tail(3) = cov_b_inv*mean_b + ((1-alpha)/(sigma*sigma)*I);
    return res;
}

/**
 * Calculates alpha for a pixel. 
 * Converges when alpha stabilizes 
 * @param alpha_guess - initial guess for alpha from trimap or previous frame
 * @param sigma deviation from matting equation
 * @param pixel with 3 [0, 1] normalized channels
 * @return calulated alpha for this pixel
 */
float calc_alpha_pixel(float const& alpha_guess, float const& sigma, Vec3f const& pixel) 
{
    float next_alpha = alpha_guess + 2*alpha_delta;
    float alpha = alpha_guess;

    VectorXf FB(6); 

    Vector3f I(pixel[0], pixel[1], pixel[2]);
    while (fabs(next_alpha - alpha) > alpha_delta) 
    {
        alpha = next_alpha;
        MatrixXf a = A(alpha, sigma);
        VectorXf y = Y(alpha, sigma, I);
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

/**
 * Calculates alpha map for a frame
 * @param trimap initial guess for alpha from trimap or previous frame alpha
 * @param sigma deviation from matting equation
 * @param img Mat with frame, 3 cahnnels [0, 1] normalized
 * @return Mat with alpha map
 */
Mat bayes_alpha_map(Mat trimap, Mat  img, double  sigma)
{
    Mat alpha_map = Mat::zeros(img.size(), CV_32F);
    int counter = 0;
    float progress = 0.0;
    for(int i=0; i<img.rows; i++) 
    {
        for(int j=0; j<img.cols; j++) 
        {   

            Vec3f cvI = img.at<Vec3f>(i,j);
            float alpha_guess = trimap.at<float>(i, j);
            float alpha;

            alpha = calc_alpha_pixel(alpha_guess, sigma, cvI);

            alpha_map.at<float>(i, j) = alpha;
            
            progress = 100.0*counter/(img.rows*img.cols);
            if (fmod(progress, 2.0) == 0) 
            {
                cout << "               \r"<< progress  << "%" <<flush;
            }
            counter ++;
            
        }        
    }
    cout << endl;
    return alpha_map;
}



int main(int argc, char *argv[]) 
{
    const string source = argv[1];  
    const string trimap_path = argv[2];        
    const string key_frame_path = argv[3];
    
    const float sigma = atof(argv[4]);    

    Mat key_frame = imread(key_frame_path, IMREAD_COLOR);
    key_frame.convertTo(key_frame, CV_32FC3, 1.0/255);

    Mat trimap = imread(trimap_path, IMREAD_GRAYSCALE);

    cout << "Processing trimap" << endl;
    auto tri_foreground = extract_pixels(key_frame, trimap, 255);
    auto tri_background = extract_pixels(key_frame, trimap, 0);

    cout << "Found " << tri_foreground[0].size() << " pixels marked as foreground by trimap" << endl;
    cout << "Found " << tri_background[0].size() << " pixels marked as background by trimap" << endl;

    auto gauss_bg = fit_gauss(tri_background);
    auto gauss_fg = fit_gauss(tri_foreground);

    mean_f = get<0>(gauss_fg);
    mean_b = get<0>(gauss_bg);
    cov_f_inv = get<1>(gauss_fg);
    cov_b_inv = get<1>(gauss_bg);
    
    cout << "Foreground mu " << endl << mean_f << endl;
    cout << "Foreground inverted covariance " << endl << cov_f_inv << endl;
    cout << "Background mu " << endl << mean_b << endl;
    cout << "Background inverted covariance " << endl << cov_b_inv << endl;

    VideoCapture sourceVideo(source);              
    if (!sourceVideo.isOpened())
    {
        cout  << "Could not open the input video: " << source << endl;
        return -1;
    }
    Size S = Size((int) sourceVideo.get(CAP_PROP_FRAME_WIDTH),   
                  (int) sourceVideo.get(CAP_PROP_FRAME_HEIGHT));
    int ex = static_cast<int>(sourceVideo.get(CAP_PROP_FOURCC));


    

    string::size_type pAt = source.find_last_of('.');                
    const string outputFileName = source.substr(0, pAt) + "_alpha.avi";
    VideoWriter alphaVideo;                                       
    alphaVideo.open(outputFileName, ex, sourceVideo.get(CAP_PROP_FPS), S, false);
    if (!alphaVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << outputFileName << endl;
        return -1;
    }

    int frame_counter = 0;
    Mat frame;
    trimap.convertTo(trimap, CV_32F, 1.0/255);

    for(;;) 
    {   
        cout << "Processing frame #" << frame_counter << endl;

        sourceVideo >> frame;
        if (frame.empty()) break;
        frame.convertTo(frame, CV_32F, 1.0/255);

        // bayess algo using manual trimap for first frame
        // and alpha map computed for frame #1 as trimap for frame #2
        Mat alpha_map = bayes_alpha_map(trimap, frame, sigma);
        trimap = alpha_map.clone();
        
        alpha_map.convertTo(alpha_map, CV_8U, 255);
        alphaVideo << alpha_map;

        frame_counter ++;
        imwrite("alpha_test1.png", alpha_map);
        break;
    }

}