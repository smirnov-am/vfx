#include <iostream> 
#include <opencv2/core.hpp>   
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;


int main(int argc, char *argv[]) {
    const string source = argv[1];          
    const string alpha_map = argv[2];
    const string background = argv[3];

    VideoCapture sourceVideo(source);              
    if (!sourceVideo.isOpened())
    {
        cout  << "Could not open the input video: " << source << endl;
        return -1;
    }
    Size S = Size((int) sourceVideo.get(CAP_PROP_FRAME_WIDTH),   
                  (int) sourceVideo.get(CAP_PROP_FRAME_HEIGHT));
    int ex = static_cast<int>(sourceVideo.get(CAP_PROP_FOURCC));

    VideoCapture alphamapVideo(alpha_map);              
    if (!alphamapVideo.isOpened())
    {
        cout  << "Could not open the alpha_map video: " << alpha_map << endl;
        return -1;
    }   

    VideoCapture bgVideo(background);              
    if (!bgVideo.isOpened())
    {
        cout  << "Could not open the background video: " << background << endl;
        return -1;
    }  

    string::size_type pAt = source.find_last_of('.');                
    const string outputFileName = source.substr(0, pAt) + "_output.avi";
    VideoWriter outputVideo;                                       
    outputVideo.open(outputFileName, ex, sourceVideo.get(CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << outputFileName << endl;
        return -1;
    }

    int frame_count = 0;
    Mat frame, background_frame, alpha_frame, res, res_front, res_back;
    vector<Mat> frame_channels, background_frame_channels, alpha_channels;
    for(;;) 
    {   
        cout << "Processing frame #" << frame_count << endl;
        sourceVideo >> frame;
        if (frame.empty()) break;
        frame.convertTo(frame, CV_32F, 1.0/255);


        bgVideo >> background_frame;
        if (background_frame.empty()) break;
        background_frame.convertTo(background_frame, CV_32F, 1.0/255);

        alphamapVideo >> alpha_frame;
        if (alpha_frame.empty()) break;
        alpha_frame.convertTo(alpha_frame, CV_32F, 1.0/255);

        split(frame, frame_channels);
        split(background_frame, background_frame_channels);
        split(alpha_frame, alpha_channels);

        for (int i=0; i < 3; ++i) 
        {
            multiply(alpha_channels[0], frame_channels[i], frame_channels[i]);
            multiply(Scalar::all(1.0) - alpha_channels[0], background_frame_channels[i], background_frame_channels[i]);
        }
        merge(frame_channels, res_front);
        merge(background_frame_channels, res_back);
        
        res = res_front + res_back;
        res.convertTo(res, CV_8UC3, 255);
        outputVideo << res;
        
        frame_count ++;
    }

}