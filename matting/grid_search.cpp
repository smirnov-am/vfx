#include <iostream> 
#include <string>
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>
#include "matting.h"


using namespace std;
using namespace cv;

int main(int argc, char *argv[]) 
{
    const string trimap_path = argv[1];        
    const string key_frame_path = argv[2];


    Mat key_frame = imread(key_frame_path, IMREAD_COLOR);
    key_frame.convertTo(key_frame, CV_32FC3, 1.0/255);

    Mat trimap = imread(trimap_path, IMREAD_GRAYSCALE);
    trimap.convertTo(trimap, CV_32F, 1.0/255);

    for (int hidden_layer=25;hidden_layer<100;hidden_layer+=10)
    {
        for (float eps=0.01;eps<1;eps+=0.1)
        {
            for (int iter=1000;iter<10000;iter+=1000)
            {
                MlpMatting mlp_matter(hidden_layer, eps, iter, trimap);
                mlp_matter.model(key_frame);

                Mat alpha_map = mlp_matter.alpha_map(key_frame);
        
                alpha_map.convertTo(alpha_map, CV_8U, 255);
                const string outputFileName = "mlp_alpha_" + to_string(hidden_layer)+ "_" +
                                                             to_string(eps)+ "_" +
                                                             to_string(iter)+ ".png";
                imwrite(outputFileName, alpha_map);

            }
        }
    }
}