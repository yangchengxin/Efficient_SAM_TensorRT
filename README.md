<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <h3 align="center">⌨️ Let us use the efficient sam！</h3>
</p>

<!-- markdownlint-enable MD033 -->
## ⚡ Enviroments
Generate the executable demo by CMakeLists.txt
1. TensorRT-8.6.1.6
2. OpenCV-4.5.5
3. CUDA-11.7

## ⚙ pt model -> engine
run the onnx2trt.cpp demo to convert the torch model(pt) to tensorrt model(engine).
encoder model and decoder model should be requested to convert to engine model.

## 🏃‍♂️ Run 
You should set the path of both encoder and decoder model, and set the direction of input image. Then, you can run the main.cpp
to segment anything.

