#include <iostream>
#include "config.h"
#include "sam.h"

using namespace std;

int main(int argc, const char* argv[])
{
	//int argc = 5;
	//const char* argv[100] = { "EfficientVIT_TRT.exe", "--points", "320,240,100,100,", "--image", "cat.jpg" };
	//const char* argv[100] = { "EfficientVIT_TRT.exe", "--boxes", "200,120,520,380,", "--image", "bear.jpeg" };

	const string encoder_path = "/home/mw/effsam/model/l1_encoder.engine";
	const string decoder_path = "/home/mw/effsam/model/l1_decoder.engine";
	EFF_VIT eff_vit(encoder_path, decoder_path, 512);

	//if (!eff_vit.parse_args(argc, argv))
	//{
	//	std::cerr << "Arguments not right!" << std::endl;
	//	std::cerr << "for example:[EfficientVIT_TRT.exe --points 320,240, --image cat.jpg]" << std::endl;
	//	return -1;
	//}
	
	//初始化模型
	eff_vit.initial_model();
	//读取输入数据并前处理
	cv::Mat input = cv::imread("/home/mw/effsam/cat.jpg");
	cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
	eff_vit.Infer_mode = "--boxes";
	while (true)
	{
		auto start_ = std::chrono::system_clock::now();
		if (eff_vit.Infer_mode == "--boxes")
		{
			eff_vit.Boxes_Infer(input);
		}
		else if (eff_vit.Infer_mode == "--points")
		{
			eff_vit.Infer(input);
		}
		auto end_ = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end_ - start_;
		cout << "infer and process cost time : " << diff.count() * 1000 << "ms" << endl;
	}
	bool suc = eff_vit.Release();

	system("pause");
	return 0;
}