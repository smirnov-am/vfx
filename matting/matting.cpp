#include <iostream> 
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>
#include "bayes.h"


using namespace std;
using namespace cv;
using namespace Eigen;


int main(int argc, char *argv[]) 
{
    const string source = argv[1];  
    const string trimap_path = argv[2];        
    const string key_frame_path = argv[3];
    
    Mat key_frame = imread(key_frame_path, IMREAD_COLOR);
    key_frame.convertTo(key_frame, CV_32FC3, 1.0/255);

    Mat trimap = imread(trimap_path, IMREAD_GRAYSCALE);
    trimap.convertTo(trimap, CV_32F, 1.0/255);


    BayesMatting matter(0.01, 0.1, trimap);
    matter.model(key_frame);


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
    alphaVideo.open(outputFileName, ex, sourceVideo.get(CAP_PROP_FPS), S, false); // last false is for greyscale video
    if (!alphaVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << outputFileName << endl;
        return -1;
    }

    int frame_counter = 0;
    Mat frame;

    for(;;) 
    {   
        cout << "Processing frame #" << frame_counter << endl;

        sourceVideo >> frame;
        if (frame.empty()) break;
        frame.convertTo(frame, CV_32F, 1.0/255);

        Mat alpha_map = matter.alpha_map(frame);
        
        alpha_map.convertTo(alpha_map, CV_8U, 255);
        alphaVideo << alpha_map;

        frame_counter ++;

        // uncomment to get alpha for the first frame and avoid waiting 
        //imwrite("alpha_oop.png", alpha_map);
        //break;
    }
}