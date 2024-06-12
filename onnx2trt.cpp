#include "int8.h"
#include <NvInferPlugin.h>

//#define FP16 false

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr) {
	return shared_ptr<_T>(ptr, [](_T* p) {p->destroy(); });
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

int onnxToTRTModel(const char* onnx_path, const char* engine_path, cv::String calib_folder, bool INT8, bool FP16)
{
	//注册插件表
	// nvinfer1::ILogger* _gLogger = NULL;
	// initLibNvInferPlugins(_gLogger, "");

	TRTLogger gLogger;
	auto builder = nvinfer1::createInferBuilder(gLogger);

	//tensorRT支持两种指定网络的模式：显示批处理和隐式批处理。（隐式批处理是早期版本使用，现在已经弃用）
	//我们使用的显示批处理的指定定义方式：
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	auto parser = nvonnxparser::createParser(*network, gLogger);

	bool parser_status = parser->parseFromFile(onnx_path, 3);
	nvinfer1::Dims input0_dim = network->getInput(0)->getDimensions();
	nvinfer1::Dims input1_dim = network->getInput(1)->getDimensions();
	nvinfer1::Dims input2_dim = network->getInput(2)->getDimensions();

	//如果是动态batch的情况下
	if (input0_dim.d[0] == -1)
	{
		const char* name = network->getInput(0)->getName();
		nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
		profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input0_dim.d[1], input0_dim.d[2], input0_dim.d[3]));
		profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(4, input0_dim.d[1], input0_dim.d[2], input0_dim.d[3]));
		profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(9, input0_dim.d[1], input0_dim.d[2], input0_dim.d[3]));
		config->addOptimizationProfile(profile);
	}
	else if (input1_dim.d[0] == -1 && input1_dim.d[1] == -1 && input2_dim.d[0] == -1 && input2_dim.d[1] == -1)
	{
		const char* name_1 = network->getInput(1)->getName();
		const char* name_2 = network->getInput(2)->getName();

		nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

		profile->setDimensions(name_1, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, input1_dim.d[2]));
		profile->setDimensions(name_1, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(4, 4, input1_dim.d[2]));
		profile->setDimensions(name_1, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(9, 9, input1_dim.d[2]));

		profile->setDimensions(name_2, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
		profile->setDimensions(name_2, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(4, 4));
		profile->setDimensions(name_2, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(9, 9));
		config->addOptimizationProfile(profile);
	}

	//FP16 or INT8 config set
	config->setMaxWorkspaceSize(1 << 30);

	//满足设备能力和量化需求的情况下
	if (builder->platformHasFastInt8() && INT8 == true)
	{
		std::cout << "support int8 and doing" << std::endl;
		vector<cv::String> calib_files;

		/********************指定校准数据集的路径********************/
		cv::String folder = calib_folder;

		std::vector<cv::String> paths;
		//cv::glob(folder, paths);
		//实例化一个校准器，参数为vector<string>的图像路径数组，和nvinfer1::Dims的维度
		//auto* calib = new Int8MINMAXCalibrator(paths, dim);

		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		//config->setInt8Calibrator(calib);
	}
	else if (FP16)
	{
		std::cout << "support Fp16 and doing" << std::endl;
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	else
	{
		std::cout << "support Fp32 and doing" << std::endl;
	}

	//if(true)
	//{
	//	config->setFlag(nvinfer1::BuilderFlag::)
	//}

	//serialize
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	nvinfer1::IHostMemory* modelstream{ nullptr };
	assert(engine != nullptr);
	modelstream = engine->serialize();

	//write to file
	std::ofstream p(engine_path, std::ios::binary);
	if (!p)
	{
		std::cerr << "could not open output file to save model" << std::endl;
		return false;
	}
	p.write(reinterpret_cast<const char*>(modelstream->data()), modelstream->size());
	std::cout << "generate file success" << std::endl;

	//Release resources
	modelstream->destroy();
	network->destroy();
	engine->destroy();
	builder->destroy();
	config->destroy();
	return 0;
}

int onnx_to_engin()
{
	//1.构建builder，并使用builder构建Network用于存储模型信息
	//2.使用Network构建parser用于从onnx文件中解析模型信息并回传给Network
	//3.使用builder构建profile用于设置动态维度，并从dynamicBinding中获取动态维度信息
	//4.构建calibrator用于校准模型，并通过BatchStream加载校准数据集
	//5.使用Builder构建Config用于设置生成Engine的参数，包括Calibrator和Profile
	//6.Builder使用Network中的模型信息和Config中的参数来生成Engine以及校准参数calParameter
	//7.通过BatchStream加载待测试数据集并传入engine，最终输出结果

	const char* onnx_model = "/home/mw/effsam/model/l1_encoder.onnx";
	const char* engin_model = "/home/mw/effsam/model/l1_encoder.engine";
	// Create builder
	TRTLogger gLogger;
	auto builder = createInferBuilder(gLogger);

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	IBuilderConfig* config = builder->createBuilderConfig();

	//Calibrator = IInt8EntropyCalibrator2::IInt8Calibrator

	//if (builder->platformHasFastInt8())
	//{
	//	config->setFlag(BuilderFlag::kINT8);
	//	config->setInt8Calibrator(Calibrator.get());
	//	config->setProfileStream()
	//}
	//else
	//{
	//	cout << "你的设备不支持int8推理和量化" << endl;
	//}


	// Create model to populate the network
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	// Parse ONNX file
	auto parser = nvonnxparser::createParser(*network, gLogger);

	bool parser_status = parser->parseFromFile(onnx_model, 3);

	// Get the name of network input
	Dims dim = network->getInput(0)->getDimensions();

	if (dim.d[0] == -1)  // -1 means it is a dynamic model
	{
		const char* name = network->getInput(0)->getName();
		IOptimizationProfile* profile = builder->createOptimizationProfile();
		profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(4, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(20, dim.d[1], dim.d[2], dim.d[3]));
		config->addOptimizationProfile(profile);
		//config->setCalibrationProfile()
	}

	// Build engine
	config->setMaxWorkspaceSize(1 << 30);
	if (false) {
		config->setFlag(nvinfer1::BuilderFlag::kFP16); // 设置精度计算
		//config->setFlag(nvinfer1::BuilderFlag::kINT8);
	}

	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	//builder->buildSerializedNetwork()
	// Serialize the model to engine file
	IHostMemory* modelStream{ nullptr };
	assert(engine != nullptr);
	modelStream = engine->serialize();

	std::ofstream p(engin_model, std::ios::binary);
	if (!p) {
		std::cerr << "could not open output file to save model" << std::endl;
		return -1;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	std::cout << "generate file success!" << std::endl;

	// Release resources
	modelStream->destroy();
	network->destroy();
	engine->destroy();
	builder->destroy();
	config->destroy();
	return 0;
}

int main()
{
	const char* onnx_path = "/home/mw/effsam/model/l1_decoder.onnx";
	const char* engine_path = "/home/mw/effsam/model/l1_decoder.engine";
	const cv::String calib_path = "";
	bool int8 = false;
	bool fp16 = true;
	// int suc = onnxToTRTModel(onnx_path, engine_path, calib_path, int8, fp16);
    onnx_to_engin();

	system("pause");
	return 0;
}