#pragma once
#include <opencv2/core.hpp> 
#include <eigen/Eigen/Dense>
#include <opencv2/ml/ml.hpp>

class Matting 
{
    public:
        /**
         * Call this method before generating alpha maps.
         * It analyzes a keyframe and trains neural network or builds a prior
         * @param keyframe opencv mat with keyframe
         */
        virtual void model(cv::Mat keyframe) {};

        /**
         * Calcualtes alpha map for a video frame. 
         * Feed consecutive frames to it 
         * @param frame opencv mat with a frame
         */
        virtual cv::Mat alpha_map(cv::Mat frame) { cv::Mat m; return m;};
};

class MlpMatting: public Matting {
        int hidden_layer_size;

        float epsilon; // the desired accuracy or change in parameters at which the iterative algorithm stops
        int max_iter; // deviation from matting equation
        cv::Mat trimap; // holds initial manual trimap

        cv::Ptr<cv::ml::ANN_MLP> mlp; // NN model

    public:
        MlpMatting(int hidden_layer_size, float epsilon, int max_iter, cv::Mat trimap): 
            hidden_layer_size(hidden_layer_size), 
            epsilon(epsilon), 
            max_iter(max_iter), 
            trimap(trimap) {};
        /**
         * Call this method before generating alpha maps.
         * It analyzes a keyframe and trains neural network
         * @param keyframe opencv mat with keyframe
         */
        virtual void model(cv::Mat keyframe);

        /**
         * Calcualtes alpha map for a video frame. 
         * Feed consecutive frames to it 
         * @param frame opencv mat with a frame
         */
        virtual cv::Mat alpha_map(cv::Mat frame);
};


class BayesMatting: public Matting {
    private:
        Eigen::MatrixXf cov_f_inv, cov_b_inv;

        Eigen::Vector3f mu_f, mu_b;
        float epsilon; // alpha convergance error
        float sigma; // deviation from matting equation
        cv::Mat trimap; // holds initial manual trimap

        /**
         * Populates A in Ax=Y equation
         * @param alpha value for a pixel
         * @param sigma deviation from matting equation
         * @return Eigen 6x6 matrix
         */ 
        Eigen::MatrixXf A(float const& alpha);

        /**
         * Populates Y in Ax=Y equation
         * @param alpha value for a pixel
         * @param sigma deviation from matting equation
         * @param I pixel with 3 [0, 1] normalized channels
         * @return Eigen 6x1 vector
         */
        Eigen::VectorXf Y(float const& alpha, Eigen::Vector3f const& I);

        /**
         * Calculates alpha for a pixel. 
         * Converges when alpha stabilizes 
         * @param alpha_guess - initial guess for alpha from trimap or previous frame
         * @param sigma deviation from matting equation
         * @param pixel with 3 [0, 1] normalized channels
         * @return calulated alpha for this pixel
         */
        float calc_alpha_pixel(float const& alpha_guess, cv::Vec3f const& pixel);

    public:
        BayesMatting(float epsilon, float sigma, cv::Mat trimap): epsilon(epsilon), sigma(sigma), trimap(trimap) {};
        
        /**
         * Call this method before generating alpha maps.
         * It analyzes a keyframe and builds a prior
         * @param keyframe opencv mat with keyframe
         */
        virtual void model(cv::Mat keyframe);

        /**
         * Calcualtes alpha map for a video frame. 
         * Feed consecutive frames to it 
         * @param frame opencv mat with a frame
         */
        virtual cv::Mat alpha_map(cv::Mat frame);
};

class MatterContext 
{
    private:
        Matting* m;
    public:
       MatterContext(){}
       void model(cv::Mat keyframe) {m->model(keyframe);}
       cv::Mat alpha_map(cv::Mat frame) {return m->alpha_map(frame);}
       void set_algo(Matting* concrete_matter) {m=concrete_matter;}
};