#include <iostream> // for standard I/O
#include <string>   // for strings
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat)
#include <opencv2/videoio.hpp>  // Video write
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // parse input arguments
    if (argc != 5)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }

    const string source = argv[1];          
    const string background = argv[2];
    const float a1 = atof(argv[3]);
    const float a2 = atof(argv[4]);
    
    // open source video
    VideoCapture inputVideo(source);              
    if (!inputVideo.isOpened())
    {
        cout  << "Could not open the input video: " << source << endl;
        return -1;
    }
    Size S = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),   
                  (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));
   
    // open background image
    Mat img = imread(background, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << background << endl;
        return 1;
    }
    img.convertTo(img, CV_32FC3, 1.0/255);
    
    //output video
    string::size_type pAt = source.find_last_of('.');                
    const string outputFileName = source.substr(0, pAt) + "_with_bg.avi";
    VideoWriter outputVideo;                                       
    outputVideo.open(outputFileName, ex, inputVideo.get(CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << outputFileName << endl;
        return -1;
    }
   
    // processing
    Mat src, res_front, res_back, res, alpha_res;
    
    vector<Mat> channels, img_channels;
    
    Mat alpha = Mat::zeros(S, CV_32F);
    for(;;) 
    {
        inputVideo >> src;    
        
        if (src.empty()) break;
        src.convertTo(src, CV_32F, 1.0/255);
        
        split(src, channels);
        split(img, img_channels);
        
        // alpha Vlahos form
        alpha = Scalar::all(1.0) - a1*(channels[1] - a2*channels[0]);

        // keep alpha in [0, 1] range
        threshold(alpha, alpha, 1, 1, THRESH_TRUNC);
        threshold(-1*alpha, alpha, 0, 0, THRESH_TRUNC);
        alpha = -1 * alpha;
        
        // applying alpha
        for (int i=0; i < 3; ++i) {
            multiply(alpha, channels[i], channels[i]);
            multiply(Scalar::all(1.0) - alpha, img_channels[i], img_channels[i]);
            
        }
            
        merge(channels, res_front);
        merge(img_channels, res_back);
        
        res = res_front + res_back;
        outputVideo << res;
    }
    return 0;
}