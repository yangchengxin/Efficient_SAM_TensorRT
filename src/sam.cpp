#include "sam.h"
#include "config.h"
#include <NvInferPlugin.h>

//vector<float> pre_points = {};
//vector<vector<float>>points = { {320,240} ,{400,240}  };
//vector<vector<float>>points = {};

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t) {
	switch (t) {
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR:   return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO:    return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknow";
	}
}

class TRTLogger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			// 打印带颜色的字符，格式如下：
			// printf("\033[47;33m打印的文本\033[0m");
			// 其中 \033[ 是起始标记
			//      47    是背景颜色
			//      ;     分隔符
			//      33    文字颜色
			//      m     开始标记结束
			//      \033[0m 是终止标记
			// 其中背景颜色或者文字颜色可不写
			// 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
			if (severity == Severity::kWARNING) {
				printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else if (severity <= Severity::kERROR) {
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else {
				printf("%s: %s\n", severity_string(severity), msg);
			}
		}
	}
};

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr) {
	return shared_ptr<_T>(ptr, [](_T* p) {p->destroy(); });
}

void SamResize::forward()
{
	old_h = input_img.cols;
	old_w = input_img.rows;
	old_channel =  input_img.channels();
	long_side_length = max(old_h, old_w);
	if (long_side_length != img_size) {
		float scale = img_size * 1.0 / max(old_h, old_w);
		new_h = old_h * scale; new_w = old_w * scale;
		new_w = int(new_w + 0.5);
		new_h = int(new_h + 0.5);
		new_wh.push_back(new_h);
		new_wh.push_back(new_w);
		vector<int> target_size = new_wh;
		cv::resize(input_img, input_img, cv::Size(target_size[0], target_size[1]));
	}
}

void EFF_VIT::preprocess(cv::Mat input)
{
	SamResize samresize(input, Img_size);
	samresize.forward();
	cv::Mat post_input = samresize.input_img;
	//post_input = post_input / 255.0f;

	int width_, height_, channel_;
	width_ = post_input.cols;
	height_ = post_input.rows;
	channel_ = post_input.channels();
	int area_ = width_ * height_;

	post_input.convertTo(post_input, CV_32F, 1.0 / 255.0f);
	
	uchar* pimage = post_input.data;

	for (int c = 0; c < channel_; c++)
	{
		for (int h = 0; h < height_; h++)
		{
			for (int w = 0; w < width_; w++)
			{
				post_input.ptr<float>(h, w)[c] = float(post_input.ptr<float>(h, w)[c] - float(pixel_mean[c])) / float(pixel_std[c]);
			}
		}
	}

	//pad->(512 * 512) or (1024 * 1024)
	cv::Mat Pad(cv::Size(Img_size, Img_size), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Rect roi(0, 0, width_, height_);
	cv::Mat roi_region = Pad(roi);
	post_input.copyTo(roi_region);

	TRT_Input = Pad;
}

void EFF_VIT::malloc_data(nvinfer1::ICudaEngine* engine)
{
	/*
	* engine->getNbIOTensors()    返回模型的输入和输出索引总数
	* engine->getNbBindings()     返回模型的输入和输出索引总数
	* engine->getBindingIndex()   输入：const char* 类型的名称；返回int32_t 类型的索引（我们想关联的张量的索引）
	* engine->getIOTensorName()   输入：int32_t 类型的索引； 返回：const char* 的输入（输出）张量的名称
	* engine->getTensorShape()    输入：const char* 类型的名称；返回nvinfer1::Dims 的维度数据。
	* engine->getBindingDimensions()  输入：int32_t 类型的索引；返回nvinfer1::Dims 的维度数据。
	* context->setBindingDimensions()  输入：int32_t 类型的索引和Dims数据；
	* context->setBindingDimensions()  输入：const char* 类型的名称和Dims数据；
	*/
	const int IO_nb = engine->getNbIOTensors();
	int en_InputIdx = 0;
	int en_OutputIdx = 0;
	int de_InputIdx1 = 0;
	int de_InputIdx2 = 0;
	int de_InputIdx3 = 0;
	int de_OutputIdx1 = 0;
	int de_OutputIdx2 = 0;
	switch (IO_nb)
	{
		case 2:
		{
			std::cout << "mallocing encoder engine data" << endl;
			/*输入输出的名称与索引关联*/
			en_InputIdx = engine->getBindingIndex(encoder_input_name);
			en_OutputIdx = engine->getBindingIndex(encoder_output_name);
			nvinfer1::Dims input_dim = engine->getTensorShape(encoder_input_name);
			nvinfer1::Dims output_dim = engine->getTensorShape(encoder_output_name);

			//计算输入和输出的维度
			for (int ii = 0; ii < input_dim.nbDims; ii++)
			{
				encoder_input_numel *= input_dim.d[ii];
			}
			for (int oo = 0; oo < output_dim.nbDims; oo++)
			{
				encoder_output_numel *= output_dim.d[oo];
			}

			//给输入输出分配主机和gpu存储空间
			checkRuntime(cudaMallocHost(&encoder_input_host_data, encoder_input_numel * sizeof(float)));
			checkRuntime(cudaMalloc(&encoder_input_device_data, encoder_input_numel * sizeof(float)));
			checkRuntime(cudaMallocHost(&encoder_output_host_data, encoder_output_numel * sizeof(float)));
			checkRuntime(cudaMalloc(&encoder_output_device_data, encoder_output_numel * sizeof(float)));
		}
		break;
		case 5:
		{
			std::cout << "macllocing decoder engine data" << endl;
			/*输入输出的名称与索引关联*/
			de_InputIdx1 = engine->getBindingIndex(decoder_input_name1);
			de_InputIdx2 = engine->getBindingIndex(decoder_input_name2);
			de_InputIdx3 = engine->getBindingIndex(decoder_input_name3);
			de_OutputIdx1 = engine->getBindingIndex(decoder_output_name1);
			de_OutputIdx2 = engine->getBindingIndex(decoder_output_name2);
			input1_dim = engine->getTensorShape(decoder_input_name1);
			input2_dim = engine->getTensorShape(decoder_input_name2);
			input3_dim = engine->getTensorShape(decoder_input_name3);
			nvinfer1::Dims output1_dim = engine->getTensorShape(decoder_output_name1);
			nvinfer1::Dims output2_dim = engine->getTensorShape(decoder_output_name2);
			int points_num = points.size();
			int boxes_num = boxes.size();
			input2_dim.d[0] = points_num == 0 ? boxes_num : 1;
			input2_dim.d[1] = points_num == 0 ? 2 : points_num;
			input3_dim.d[0] = points_num == 0 ? boxes_num : 1;
			input3_dim.d[1] = points_num == 0 ? 2 : points_num;
			output1_dim.d[0] = points_num == 0 ? boxes_num : 1;
			output2_dim.d[0] = points_num == 0 ? boxes_num : 1;
			
			for (int ii_1 = 0; ii_1 < input1_dim.nbDims; ii_1++)
			{
				decoder_input1_numel *= input1_dim.d[ii_1];
			}
			for (int ii_2 = 0; ii_2 < input2_dim.nbDims; ii_2++)
			{
				decoder_input2_numel *= input2_dim.d[ii_2];
			}
			for (int ii_3 = 0; ii_3 < input3_dim.nbDims; ii_3++)
			{
				decoder_input3_numel *= input3_dim.d[ii_3];
			}
			for (int oo_1 = 0; oo_1 < output1_dim.nbDims; oo_1++)
			{
				decoder_output1_numel *= output1_dim.d[oo_1];
			}
			for (int oo_2 = 0; oo_2 < output2_dim.nbDims; oo_2++)
			{
				decoder_output2_numel *= output2_dim.d[oo_2];
			}
			checkRuntime(cudaMallocHost(&decoder_input1_host_data, decoder_input1_numel * sizeof(float)));
			checkRuntime(cudaMalloc(&decoder_input1_device_data, decoder_input1_numel * sizeof(float)));
			checkRuntime(cudaMallocHost(&decoder_input2_host_data, decoder_input2_numel * sizeof(float)));
			checkRuntime(cudaMalloc(&decoder_input2_device_data, decoder_input2_numel * sizeof(float)));
			checkRuntime(cudaMallocHost(&decoder_input3_host_data, decoder_input3_numel * sizeof(float)));
			checkRuntime(cudaMalloc(&decoder_input3_device_data, decoder_input3_numel * sizeof(float)));
			checkRuntime(cudaMallocHost(&decoder_output1_host_data, decoder_output1_numel * sizeof(float)));
			checkRuntime(cudaMalloc(&decoder_output1_device_data, decoder_output1_numel * sizeof(float)));
			checkRuntime(cudaMallocHost(&decoder_output2_host_data, decoder_output2_numel * sizeof(float)));
			checkRuntime(cudaMalloc(&decoder_output2_device_data, decoder_output2_numel * sizeof(float)));
		}
		break;
	}
}

nvinfer1::IExecutionContext* EFF_VIT::initial_engine(const string model)
{
	// nvinfer1::ILogger* gLogger = NULL;
	// initLibNvInferPlugins(gLogger, "");

	TRTLogger logger;

	ifstream fs(model, std::ios::binary);
	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(fs), {});
	fs.close();

	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine((void*)buffer.data(), buffer.size());

	// malloc_data (写一个判断语句 来为encoder和decoder的输入输出分配内存)
	checkRuntime(cudaStreamCreate(&stream_));
	auto context = engine->createExecutionContext();
	malloc_data(engine);
	if (engine->getNbIOTensors() == 5)
	{
		nvinfer1::Dims input1_dim_ = engine->getTensorShape(decoder_input_name1);
		nvinfer1::Dims input2_dim_ = engine->getTensorShape(decoder_input_name2);
		nvinfer1::Dims input3_dim_ = engine->getTensorShape(decoder_input_name3);
		/*nvinfer1::IExecutionContext::setBindingDimension(bindingIndex, dims)*/
		/*nvinfer1::IExecutionContext::setInputShape(tensorname, dims)---the newest api!!!*/
		/*batch 推理时，指定好engine的输入维度之后，一定要记得让context再指定一下输入维度，切记！！！*/
		context->setInputShape(decoder_input_name2, input2_dim);
		context->setInputShape(decoder_input_name3, input3_dim);
	}

	return context;
}

bool EFF_VIT::parse_args(int argc, const char* argv[])
{
	if (argc < 5) return false;
	if (std::string(argv[1]) == "--points" && std::string(argv[3]) == "--image" && argc == 5)
	{
		int cnt_int = 0;
		const char* points_ = argv[2];
		for (int i = 0; points_[i] != '\0'; i++)
		{
			/*说明是数字*/
			if (points_[i] >= '0' && points_[i] <= '9')
			{
				cnt_int *= 10;
				cnt_int += points_[i] - '0';
			}
			else if (points_[i] == ',')
			{
				pre_points.push_back(cnt_int);
				cnt_int = 0;
			}
		}
		Input_path = std::string(argv[4]);

		for (int pp = 0; pp < pre_points.size() / 2; pp++)
		{
			vector<float> single_point;
			for (int ppp = 0; ppp < 2; ppp++)
			{
				single_point.push_back(pre_points[pp * 2 + ppp]);
			}
			points.push_back(single_point);
		}
	}
	else if (std::string(argv[1]) == "--boxes" && std::string(argv[3]) == "--image" && argc == 5)
	{
		int cnt_int = 0;
		int box_count = 0;
		int box_axis = 0;
		vector<float> box_ = {};
		const char* boxes_ = argv[2];
		for (int i = 0; boxes_[i] != '\0'; i++)
		{
			if (boxes_[i] >= '0' && boxes_[i] <= '9')
			{
				cnt_int *= 10;
				cnt_int += boxes_[i] - '0';
			}
			else if (boxes_[i] == ',')
			{
				if (box_axis == 3)
				{
					box_.push_back(cnt_int);
					boxes.push_back(box_);
					box_ = {};
					box_axis = 0;
				}
				else
				{
					box_.push_back(cnt_int);
					box_axis++;
				}
				cnt_int = 0;
			}
		}
		Input_path = std::string(argv[4]);
		Infer_mode = std::string(argv[1]);
	}
	return true;
}

void EFF_VIT::get_preprocess_shape(int old_h, int old_w, int long_side_length)
{
	float scale = long_side_length * 1.0f / max(old_h, old_w);
	float new_w = old_w * scale;
	float new_h = old_h * scale;
	new_w = int(new_w + 0.5);
	new_h = int(new_h + 0.5);
	new_wh.push_back(new_h);
	new_wh.push_back(new_w);
}

void EFF_VIT::initial_model()
{
	std::cout << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;

	thread* encoder_initial = NULL; 
	thread* decoder_initial = NULL;
	encoder_initial = new thread([&] { encoder_context = initial_engine(encoder_model); });
	decoder_initial = new thread([&] { decoder_context = initial_engine(decoder_model); });
	(*encoder_initial).join();
	(*decoder_initial).join();
	delete encoder_initial;
	delete decoder_initial;
}

void EFF_VIT::prepare_encoder_buffer()
{
	int input_cols = TRT_Input.cols;
	int input_rows = TRT_Input.rows;
	int input_channels = TRT_Input.channels();
	vector<cv::Mat> chw;
	for (int pp = 0; pp < input_channels; pp++)
	{
		chw.emplace_back(cv::Mat(cv::Size(input_cols, input_rows), CV_32FC1, encoder_input_host_data + pp * input_cols * input_rows));
	}
	cv::split(TRT_Input, chw);
	checkRuntime(cudaMemcpy(encoder_input_device_data, encoder_input_host_data, encoder_input_numel * sizeof(float), cudaMemcpyHostToDevice));
}

void EFF_VIT::prepare_decoder_buffer(cv::Mat image_embedding, float* ppp, float* lll)
{
	int decoder_input1_cols = image_embedding.cols;
	int decoder_input1_rows = image_embedding.rows;
	int decoder_input1_channels = image_embedding.channels();

	vector<cv::Mat> chw_1;
	for (int pp = 0; pp < decoder_input1_channels; pp++)
	{
		chw_1.emplace_back(cv::Mat(cv::Size(decoder_input1_cols, decoder_input1_rows), CV_32FC1, decoder_input1_host_data + pp * decoder_input1_cols * decoder_input1_rows));
	}
	cv::split(image_embedding, chw_1);


	decoder_input2_host_data = ppp;
	decoder_input3_host_data = lll;
	checkRuntime(cudaMemcpy(decoder_input1_device_data, decoder_input1_host_data, decoder_input1_numel * sizeof(float), cudaMemcpyHostToDevice));
	checkRuntime(cudaMemcpy(decoder_input2_device_data, decoder_input2_host_data, decoder_input2_numel * sizeof(float), cudaMemcpyHostToDevice));
	checkRuntime(cudaMemcpy(decoder_input3_device_data, decoder_input3_host_data, decoder_input3_numel * sizeof(float), cudaMemcpyHostToDevice));
}

vector<float> EFF_VIT::apply_coords(vector<float> coords, vector<int> original_size, vector<int> new_size)
{
	int old_h = original_size[0];
	int old_w = original_size[1];
	int new_h = new_size[0];
	int	new_w = new_size[1];
	vector<float> new_coords(2);
	//uchar pixel_value = Mat.ptr<uchar>(row)[col]
	float a1 = coords[0];
	float a2 = coords[1];
	float b1 = float(new_w) / float(old_w);
	float b2 = float(new_h) / float(old_h);
	new_coords[0] = a1 * b1;
	new_coords[1] = a2 * b2;
	return new_coords;
}

vector<float> EFF_VIT::apply_boxes(vector<float> boxes, vector<int> original_size, vector<int> new_size)
{
	/*boxes 4 -> 2,2*/
	vector<float> xy0;
	vector<float> xy1;
	vector<float> xyxy;
	for (int i = 0; i < 2; i++)
	{
		if (i == 0)
		{
			xy0.push_back(boxes[i * 2 + 0]);
			xy0.push_back(boxes[i * 2 + 1]);
			xy0 = apply_coords(xy0, original_size, new_size);
			xyxy.push_back(xy0[0]);
			xyxy.push_back(xy0[1]);
		}
		else if (i == 1)
		{
			xy1.push_back(boxes[i * 2 + 0]);
			xy1.push_back(boxes[i * 2 + 1]);
			xy1 = apply_coords(xy1, original_size, new_size);
			xyxy.push_back(xy1[0]);
			xyxy.push_back(xy1[1]);
		}
	}
	return xyxy;
}

void EFF_VIT::mask_postprocessing(cv::Mat& masks, vector<int> ori_size)
{
	int img_size = 1024;
	cv::resize(masks, masks, cv::Size(img_size, img_size), 0,0, cv::INTER_LINEAR);
	float m_scale = float(img_size) / float(max(ori_size[0], ori_size[1]));
	transformed_size.push_back(int(m_scale * ori_size[0] + 0.5));
	transformed_size.push_back(int(m_scale * ori_size[1] + 0.5));
	uchar* masks_ptr = masks.data;

	cv::Rect roi(0, 0, transformed_size[1], transformed_size[0]);
	masks = masks(roi);
}

void EFF_VIT::Infer(cv::Mat input)
{
	Input = input;
	cv::cvtColor(input, Input_BGR, cv::COLOR_RGB2BGR);
	preprocess(input); // TRT_Input

	prepare_encoder_buffer();
	float* buffers[] = { encoder_input_device_data, encoder_output_device_data };
	encoder_context->enqueueV2((void**)buffers, stream_, nullptr);
	checkRuntime(cudaMemcpy(encoder_output_host_data, encoder_output_device_data, encoder_output_numel * sizeof(float), cudaMemcpyDeviceToHost));

	//可视化encoder的输出结果
	cv::Mat image_embedding = cv::Mat(cv::Size(64, 64 * 256), CV_32FC1, encoder_output_host_data);
	/*float _p = image_embedding.at<float>(0, 3);
	cout << _p << endl;*/

	/*784-1024*/
	int H, W;
	H = Input.rows;
	W = Input.cols;

	get_preprocess_shape(H, W, 1024);
	//cout << new_wh[0] << "-" << new_wh[1] << endl;

	vector<vector<float>> point_coords;
	vector<int> ori_img_size = { H, W };
	vector<int> input_size = { new_wh[0], new_wh[1] };
	cv::Mat input_labels = cv::Mat_<float>(1, 1 * points.size());
	vector<vector<float>> input_coords;
	float points_[1][100][2] = { }; //存放处理后的points，最多存放100个点
	float labels_[1][1][100] = { }; 
	for (int vv = 0; vv < points.size(); vv++)
	{
		vector<float> new_coords(2);
		new_coords = apply_coords(points[vv], ori_img_size, input_size);
		points_[0][vv][0] = new_coords[0];
		points_[0][vv][1] = new_coords[1];
		labels_[0][0][vv] = 1;
	}
	//cv::Mat input_coords = (cv::Mat_<float>(1, 2) << new_coords[0], new_coords[1]);
	//cv::Mat input_labels = (cv::Mat_<float>(1, 1) << 1);
	float* ppp = &points_[0][0][0];
	float* lll = &labels_[0][0][0];
	prepare_decoder_buffer(image_embedding, ppp, lll);
	float* buffers_[] = { decoder_input1_device_data, decoder_input2_device_data, decoder_input3_device_data, decoder_output1_device_data, decoder_output2_device_data };
	decoder_context->enqueueV2((void**)buffers_, stream_, nullptr);
	checkRuntime(cudaMemcpy(decoder_output1_host_data, decoder_output1_device_data, decoder_output1_numel * sizeof(float), cudaMemcpyDeviceToHost));
	checkRuntime(cudaMemcpy(decoder_output2_host_data, decoder_output2_device_data, decoder_output2_numel * sizeof(float), cudaMemcpyDeviceToHost));

	cv::Mat ious = cv::Mat(cv::Size(1, 4), CV_32FC1, decoder_output2_host_data);
	cv::Mat masks = cv::Mat(cv::Size(256, 256 * 4), CV_32FC1, decoder_output1_host_data);
	uchar* mask_ptr = masks.data;
	double min_iou, max_iou;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(ious, &min_iou, &max_iou, &min_loc, &max_loc);

	cv::Mat mask1 = cv::Mat(cv::Size(256, 256), CV_32FC1, mask_ptr + sizeof(float) * 256 * 256 * int(max_loc.y));

	mask_postprocessing(mask1, ori_img_size);
	cv::resize(mask1, mask1, cv::Size(ori_img_size[1], ori_img_size[0]));

	mask1 = mask1 > Mask_Threshold;

	cv::Mat Mask_2 = cv::Mat(cv::Size(mask1.cols, mask1.rows), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat Mask_3 = cv::Mat(cv::Size(mask1.cols, mask1.rows), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat imgs[3] = { Mask_2, mask1, Mask_3 };
	cv::Mat merge_Mask;
	cv::merge(imgs, 3, merge_Mask);
	cv::addWeighted(Input_BGR, 1, merge_Mask, 0.5, 1, Input_BGR);
	
	//画点
	for (int pp = 0; pp < points.size(); pp++)
	{
		cv::Point point1(points[pp][0], points[pp][1]);
		cv::circle(Input_BGR, point1, 5, cv::Scalar(255, 0, 0), -1);
	}

	cv::namedWindow("show", cv::WINDOW_NORMAL);
	cv::imshow("show", Input_BGR);
	cv::waitKey(0);
}

void EFF_VIT::Boxes_Infer(cv::Mat input)
{
	Input = input;
	cv::cvtColor(input, Input_BGR, cv::COLOR_RGB2BGR);
	preprocess(input); // TRT_Input

	prepare_encoder_buffer();
	float* buffers[] = { encoder_input_device_data, encoder_output_device_data };

	auto start_encoder = std::chrono::system_clock::now();
	encoder_context->enqueueV2((void**)buffers, stream_, nullptr);
	auto end_encoder = std::chrono::system_clock::now();
	std::chrono::duration<double> diff_encoder = end_encoder - start_encoder;
	cout << "encoder infer costs time : " << diff_encoder.count() * 1000 << "ms" << endl;

	checkRuntime(cudaMemcpy(encoder_output_host_data, encoder_output_device_data, encoder_output_numel * sizeof(float), cudaMemcpyDeviceToHost));

	//可视化encoder的输出结果
	cv::Mat image_embedding = cv::Mat(cv::Size(64, 64 * 256), CV_32FC1, encoder_output_host_data);
	/*float _p = image_embedding.at<float>(0, 3);
	cout << _p << endl;*/

	/*784-1024*/
	int H, W;
	H = Input.rows;
	W = Input.cols;

	get_preprocess_shape(H, W, 1024);
	//cout << new_wh[0] << "-" << new_wh[1] << endl;

	vector<vector<float>> point_coords;
	vector<int> ori_img_size = { H, W };
	vector<int> input_size = { new_wh[0], new_wh[1] };
	cv::Mat input_labels = cv::Mat_<float>(1, 1 * points.size());
	vector<vector<float>> input_coords;
	float boxes_[100][2][2] = { }; //存放处理后的points，最多存放100个点
	float labels_[1][100][2] = { };
	for (int vv = 0; vv < boxes.size(); vv++)
	{
		vector<float> new_boxes(4);
		new_boxes = apply_boxes(boxes[vv], ori_img_size, input_size);
		boxes_[vv][0][0] = new_boxes[0];
		boxes_[vv][0][1] = new_boxes[1];
		boxes_[vv][1][0] = new_boxes[2];
		boxes_[vv][1][1] = new_boxes[3];
		labels_[0][vv][0] = 2.0f;
		labels_[0][vv][1] = 3.0f;
	}
	//cv::Mat input_coords = (cv::Mat_<float>(1, 2) << new_coords[0], new_coords[1]);
	//cv::Mat input_labels = (cv::Mat_<float>(1, 1) << 1);
	float* ppp = &boxes_[0][0][0];
	float* lll = &labels_[0][0][0];
	prepare_decoder_buffer(image_embedding, ppp, lll);
	float* buffers_[] = { decoder_input1_device_data, decoder_input2_device_data, decoder_input3_device_data, decoder_output1_device_data, decoder_output2_device_data };
	auto start_decoder = std::chrono::system_clock::now();
	decoder_context->enqueueV2((void**)buffers_, stream_, nullptr);
	auto end_decoder = std::chrono::system_clock::now();
	std::chrono::duration<double> diff_decoder = end_encoder - start_encoder;
	cout << "decoder infer costs time : " << diff_decoder.count() * 1000 << "ms" << endl;

	checkRuntime(cudaMemcpy(decoder_output1_host_data, decoder_output1_device_data, decoder_output1_numel * sizeof(float), cudaMemcpyDeviceToHost));
	checkRuntime(cudaMemcpy(decoder_output2_host_data, decoder_output2_device_data, decoder_output2_numel * sizeof(float), cudaMemcpyDeviceToHost));

	cv::Mat ious = cv::Mat(cv::Size(1, 4 * boxes.size()), CV_32FC1, decoder_output2_host_data);
	cv::Mat masks = cv::Mat(cv::Size(256, 256 * 4 * boxes.size()), CV_32FC1, decoder_output1_host_data);

	uchar* masks_ptr = masks.data;
	uchar* ious_ptr = ious.data;
	for (int oo = 0; oo < boxes.size(); oo++)
	{

		cv::Mat _mask_ = cv::Mat(cv::Size(256, 256 * 4), CV_32FC1, masks_ptr + sizeof(float) * 256 * 256 * 4 * oo);
		cv::Mat _iou_ = cv::Mat(cv::Size(1, 4), CV_32FC1, ious_ptr + sizeof(float) * 1 * 4 * oo);
		double min_iou, max_iou;
		cv::Point min_loc, max_loc;
		cv::minMaxLoc(_iou_, &min_iou, &max_iou, &min_loc, &max_loc);
		cv::Mat mask1 = cv::Mat(cv::Size(256, 256), CV_32FC1, _mask_.data + sizeof(float) * 256 * 256 * int(max_loc.y));
		mask_postprocessing(mask1, ori_img_size);
		cv::resize(mask1, mask1, cv::Size(ori_img_size[1], ori_img_size[0]));
		mask1 = mask1 > Mask_Threshold;

		cv::Mat Mask_2 = cv::Mat(cv::Size(mask1.cols, mask1.rows), CV_8UC1, cv::Scalar(0, 0, 0));
		cv::Mat Mask_3 = cv::Mat(cv::Size(mask1.cols, mask1.rows), CV_8UC1, cv::Scalar(0, 0, 0));
		cv::Mat imgs[3] = { Mask_2, mask1, Mask_3 };
		cv::Mat merge_Mask;
		cv::merge(imgs, 3, merge_Mask);
		cv::addWeighted(Input_BGR, 1, merge_Mask, 0.5, 1, Input_BGR);
	}

	//画框
	for (int bb = 0; bb < boxes.size(); bb++)
	{
		cv::Point x0y0(boxes[bb][0], boxes[bb][1]);
		cv::Point x1y1(boxes[bb][2], boxes[bb][3]);
		cv::rectangle(Input_BGR, x0y0, x1y1, cv::Scalar(255, 0, 255), 2, 8, 0);
	}

	cv::namedWindow("show", cv::WINDOW_NORMAL);
	cv::imshow("show", Input_BGR);
	cv::waitKey(0);
}

bool EFF_VIT::Release()
{
	encoder_input_host_data = nullptr;
	encoder_input_device_data = nullptr;
	encoder_output_host_data = nullptr;
	encoder_output_device_data = nullptr;
	decoder_input1_host_data = nullptr;
	decoder_input1_device_data = nullptr;
	decoder_input2_host_data = nullptr;
	decoder_input2_device_data = nullptr;
	decoder_input3_host_data = nullptr;
	decoder_input3_device_data = nullptr;
	decoder_output1_host_data = nullptr;
	decoder_output1_device_data = nullptr;
	decoder_output2_host_data = nullptr;
	decoder_output2_device_data = nullptr;
	encoder_context->destroy();
	decoder_context->destroy();
	cudaStreamDestroy(stream_);
	return true;
}

