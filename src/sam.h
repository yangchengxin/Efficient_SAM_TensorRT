#pragma once
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <fstream>
// #include <windows.h>
#include <vector>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInferPlugin.h>
#include <assert.h>
#include <math.h>
#include <thread>


using namespace std;
using namespace nvinfer1;


class SamResize
{
public:
	SamResize(cv::Mat _input_img, int _img_size) : img_size(_img_size), input_img(_input_img)
	{
		cout << "初始化SamResize！" << endl;
	}
	~SamResize()
	{
		cout << "析构SamResize！" << endl;
	}

	int img_size;
	cv::Mat input_img;
	void forward();

private:
	int new_h;
	int new_w;
	int old_h;
	int old_w;
	int old_channel;
	int long_side_length;
	vector<int> new_wh;
};

class EFF_VIT
{
public:
	EFF_VIT(const string encoder_path, const string decoder_path, const int img_size) : encoder_model(encoder_path), decoder_model(decoder_path), Img_size(img_size)
	{
		cout << "调用了EFF_VIT构造函数" << endl;
	}
	~EFF_VIT()
	{
		cout << "调用了EFF_VIT析构函数" << endl;
	}

	vector<float> apply_coords(vector<float> coords, vector<int> original_size, vector<int> new_size);
	void preprocess(cv::Mat coords);
	void initial_model();
	void malloc_data(nvinfer1::ICudaEngine* engine);
	nvinfer1::IExecutionContext* initial_engine(const string model);
	void Infer(cv::Mat input);
	void parse_point(const string _str);
	void get_preprocess_shape(int old_h, int old_w, int long_side_length);
	void prepare_encoder_buffer();
	void prepare_decoder_buffer(cv::Mat image_embedding, float* ppp, float* lll);
	void mask_postprocessing(cv::Mat& masks, vector<int> ori_size);
	bool parse_args(int argc, const char* argv[]);
	bool Release();

	void Boxes_Infer(cv::Mat input);
	vector<float> apply_boxes(vector<float> boxes, vector<int> original_size, vector<int> new_size);

	cv::Mat Input;
	cv::Mat Input_BGR;
	cv::Mat TRT_Input;
	std::string Input_path;
	nvinfer1::Dims input1_dim;
	nvinfer1::Dims input2_dim;
	nvinfer1::Dims input3_dim;
	vector<float> pre_points;
	vector<vector<float>>points = {}; /*{320,240}, {100,100} for points infer test*/
	vector<vector<float>>boxes = { {180,80,640,400}, {0,0,240,240}, {80,300,180,350} }; /*{180,80,640,400}, {0,0,240,240}, {80,300,180,350} for boxes infer test*/
	std::string Infer_mode;

private:
	cv::Mat _coords;
	const int Img_size;
	const string encoder_model;
	const string decoder_model;
	nvinfer1::IExecutionContext* encoder_context;
	nvinfer1::IExecutionContext* decoder_context;
	cudaStream_t stream_ = nullptr;
	int tensor_w = 512;
	int tensor_h = 512;
	int tensor_c = 3;

	/*encoder data ptr (including input and output)*/
	float* encoder_input_host_data = nullptr;
	float* encoder_input_device_data = nullptr;
	float* encoder_output_host_data = nullptr;
	float* encoder_output_device_data = nullptr;
	float* en_buffers[2] = { encoder_input_device_data, encoder_output_device_data };

	/*decoder data ptr (including input and output)*/
	float* decoder_input1_host_data = nullptr;
	float* decoder_input1_device_data = nullptr;
	float* decoder_input2_host_data = nullptr;
	float* decoder_input2_device_data = nullptr;
	float* decoder_input3_host_data = nullptr;
	float* decoder_input3_device_data = nullptr;
	float* decoder_output1_host_data = nullptr;
	float* decoder_output1_device_data = nullptr;
	float* decoder_output2_host_data = nullptr;
	float* decoder_output2_device_data = nullptr;
	float* de_buffers[5] = { decoder_input1_device_data, decoder_input2_host_data, decoder_input3_host_data, decoder_output1_device_data, decoder_output2_device_data };

	/*encoder tensor name(defined in onnx model)*/
	const char* encoder_input_name = "input_image";
	const char* encoder_output_name = "image_embeddings";

	int encoder_input_numel = 1;
	int encoder_output_numel = 1;

	/*decoder tensor name(defined in onnx model)*/
	const char* decoder_input_name1 = "image_embeddings";
	const char* decoder_input_name2 = "point_coords";
	const char* decoder_input_name3 = "point_labels";
	const char* decoder_output_name1 = "masks";
	const char* decoder_output_name2 = "iou_predictions";

	int decoder_input1_numel = 1;
	int decoder_input2_numel = 1;
	int decoder_input3_numel = 1;
	int decoder_output1_numel = 1;
	int decoder_output2_numel = 1;

	vector<int> new_wh;
	vector<int> transformed_size;
	float Mask_Threshold = 0.0f;
};