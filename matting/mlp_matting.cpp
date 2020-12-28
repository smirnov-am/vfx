#include <iostream> 
#include <opencv2/videoio.hpp>
#include <opencv2/ml/ml.hpp>
#include <cmath>
#include "trimap.h"


using namespace std;
using namespace cv;
using namespace ml;


Mat mlp_alpha_map(Mat img)
{
    Mat alpha_map = Mat::zeros(img.size(), CV_32F);
    return alpha_map;
}

int main(int argc, char *argv[]) 
{
    const string source = argv[1];  
    const string trimap_path = argv[2];        
    const string key_frame_path = argv[3];
    
    const float sigma = atof(argv[4]);    

    Mat key_frame = imread(key_frame_path, IMREAD_COLOR);
    key_frame.convertTo(key_frame, CV_32FC3);

    Mat trimap = imread(trimap_path, IMREAD_GRAYSCALE);

    cout << "Processing trimap" << endl;
    auto tri_foreground = extract_pixels(key_frame, trimap, 255);
    auto tri_background = extract_pixels(key_frame, trimap, 0);

    cout << "Found " << tri_foreground[0].size() << " pixels marked as foreground by trimap" << endl;
    cout << "Found " << tri_background[0].size() << " pixels marked as background by trimap" << endl;

    Mat mlp_input_foreground = to_mat(tri_foreground);
    Mat mlp_output_foreground = Mat::ones(tri_foreground[0].size(), 1, CV_32F);


    Mat mlp_input_background = to_mat(tri_background);
    Mat mlp_output_background = Mat::zeros(tri_background[0].size(), 1, CV_32F);

    Mat mlp_input, mlp_output;
    vconcat(mlp_input_foreground, mlp_input_background, mlp_input);
    vconcat(mlp_output_foreground, mlp_output_background, mlp_output);

    cout << "MLP Input "<<  mlp_input.rows << "x"<< mlp_input.cols << endl;
    cout << "MLP Output "<<  mlp_output.rows << "x"<< mlp_output.cols << endl;
    Ptr<ANN_MLP> mlp = ANN_MLP::create();
    Mat layersSize = Mat(3, 1, CV_16U);
    layersSize.row(0) = Scalar(mlp_input.cols);
    layersSize.row(1) = Scalar(25);
    layersSize.row(2) = Scalar(mlp_output.cols);
    mlp->setLayerSizes(layersSize);

    mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM, 1, 1);

    TermCriteria termCrit = TermCriteria(
        TermCriteria::Type::MAX_ITER + TermCriteria::Type::EPS,
        10000,
        0.0001
    );
    mlp->setTermCriteria(termCrit);

    mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP, 0.001);

    Ptr<TrainData> trainingData = TrainData::create(
        mlp_input,
        SampleTypes::ROW_SAMPLE,
        mlp_output
    );

    mlp->train(trainingData
        , ANN_MLP::TrainFlags::NO_INPUT_SCALE
        + ANN_MLP::TrainFlags::NO_OUTPUT_SCALE
    );

    cout <<"MLP trained "<<endl;

    Mat alpha_map = Mat::zeros(key_frame.size(), CV_32F);
    for(int i=0; i<key_frame.rows; i++) 
    {
        for(int j=0; j<key_frame.cols; j++) 
        {   
            float pixel_data[3] = {key_frame.at<Vec3f>(i,j)[0], key_frame.at<Vec3f>(i,j)[1], key_frame.at<Vec3f>(i,j)[2]};
            Mat pixel = Mat(1, 3, CV_32F, pixel_data);
            Mat result;
            mlp->predict(pixel, result);
            //cout << "Processing " << i << ":" << j << " " << " pixel " << pixel << " predicted alpha " << result << endl;;

            alpha_map.at<float>(i, j) = result.at<float>(0, 0);
            
            
        }        
    }
    alpha_map.convertTo(alpha_map, CV_8U, 255);
    imwrite("alpha_mlp.png", alpha_map);

/*     VideoCapture sourceVideo(source);              
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
        Mat alpha_map = mlp_alpha_map(frame);
        trimap = alpha_map.clone();
        
        alpha_map.convertTo(alpha_map, CV_8U, 255);
        alphaVideo << alpha_map;

        frame_counter ++;
        imwrite("alpha_test1.png", alpha_map);
        break;
    } */

}