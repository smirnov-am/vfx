
#include <iostream> 
#include "trimap.h"
#include "matting.h"

using namespace std;
using namespace cv;
using namespace ml;

void MlpMatting::model(Mat keyframe)
{
    cout << "Processing trimap" << endl;
    auto tri_foreground = extract_pixels(keyframe, trimap, 1.0);
    auto tri_background = extract_pixels(keyframe, trimap, 0);

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
    
    mlp = ANN_MLP::create();
    Mat layersSize = Mat(3, 1, CV_16U);
    layersSize.row(0) = Scalar(mlp_input.cols);
    layersSize.row(1) = Scalar(hidden_layer_size);
    layersSize.row(2) = Scalar(mlp_output.cols);
    mlp->setLayerSizes(layersSize);

    mlp->setActivationFunction(ANN_MLP::ActivationFunctions::RELU);

    TermCriteria termCrit = TermCriteria(
        TermCriteria::Type::MAX_ITER + TermCriteria::Type::EPS,
        max_iter,
        epsilon
    );
    mlp->setTermCriteria(termCrit);

    mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP, 0.1);

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
}

Mat MlpMatting::alpha_map(Mat frame)
{
    Mat alpha_map = Mat::zeros(frame.size(), CV_32F);
    int counter = 0;
    float progress = 0.0;

    for(int i=0; i<frame.rows; i++) 
    {
        for(int j=0; j<frame.cols; j++) 
        {   
            float pixel_data[3] = {frame.at<Vec3f>(i,j)[0], frame.at<Vec3f>(i,j)[1], frame.at<Vec3f>(i,j)[2]};
            Mat pixel = Mat(1, 3, CV_32F, pixel_data);
            Mat result;
            mlp->predict(pixel, result);

            alpha_map.at<float>(i, j) = result.at<float>(0, 0);

            
            progress = 100.0*counter/(frame.rows*frame.cols);
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