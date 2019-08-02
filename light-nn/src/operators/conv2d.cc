//头文件
#include "operators/conv2d.h"
#include "utils/math-functions.h"


namespace lnn {

	float * convert_x(const float *A, float *A_convert, const int outM, const int in_channels, int Map, int Kernel)
	{//转换被卷积矩阵 这里根据步长和kernel_size大小进行转换
		//定义被卷积矩阵宽高
		const int convAw = Kernel * Kernel * in_channels;
		const int convAh = outM * outM;
		//float A_convert[convh*convAw] = { 0 };
		for (int i = 0; i < outM; i++)
		{
			for (int j = 0; j < outM; j++)
			{
				int wh = i * outM * convAw + j * convAw;
				for (size_t k = 0; k < in_channels; k++)
				{
					//int wh = k * outM * outM *convAw+ i * outM * convAw + j * convAw;

					int col1 = k * Map * Map + i * Map + j; //一个k跨过map*map个元素，三维先拉伸成二维，再拉伸成一维
					A_convert[wh] = A[col1];
					A_convert[wh + 1] = A[col1 + 1];
					A_convert[wh + 2] = A[col1 + 2];

					int col2 = k * Map * Map + (i + 1) * Map + j;
					A_convert[wh + 3] = A[col2];
					A_convert[wh + 4] = A[col2 + 1];
					A_convert[wh + 5] = A[col2 + 2];

					int col3 = k * Map * Map + (i + 2) * Map + j;
					A_convert[wh + 6] = A[col3];
					A_convert[wh + 7] = A[col3 + 1];
					A_convert[wh + 8] = A[col3 + 2];
					wh += Kernel * Kernel;
				}
			}
		}

		return A_convert;
	}
	Conv2D::Conv2D(const Json::Value &config) : Operator(config) {
		m_name = config["name"].asString();
		m_input_size = config["param"]["input_size"].asInt();
		m_output_size = config["param"]["output_size"].asInt();
		m_kernel_size = config["param"]["kernel_size"].asInt();
		if (config["param"].isMember("stride")) {
			m_stride = config["param"]["stride"].asInt();
		}
		else {
			m_stride = 1;
		}
		if (config["param"].isMember("padding")) {
			m_padding = config["param"]["padding"].asInt();
		}
		else {
			m_padding = 0;
		}
		if (config["param"].isMember("dilation")) {
			m_dilation = config["param"]["dilation"].asInt();
		}
		else {
			m_dilation = 1;
		}
		if (config["param"].isMember("bias")) {
			b_bias = config["param"]["bias"].asBool();
		}
		else {
			b_bias = true;
		}
	}

	Conv2D::~Conv2D() {
	}

	bool Conv2D::set_weight(const std::vector<Tensor> &weights,
		const std::map<std::string, size_t> &weights_name2id) {
		std::vector<std::string> tensor_name;
		std::vector<size_t> tensor_size, shape;
		std::vector<std::vector<size_t> > tensor_shape;
		m_weights.resize(1);
		tensor_name.push_back(m_name + ".weight");
		tensor_size.push_back(m_output_size * m_kernel_size *m_kernel_size* m_input_size);
		shape.push_back(m_output_size);
		shape.push_back(m_kernel_size);
		shape.push_back(m_input_size);
		tensor_shape.push_back(shape);
		if (b_bias) {
			m_weights.resize(2);
			tensor_name.push_back(m_name + ".bias");
			tensor_size.push_back(m_output_size);
			shape.resize(1);
			tensor_shape.push_back(shape);
		}

		// get tensors needed by current operator
		std::map<std::string, size_t>::const_iterator it;
		for (size_t i = 0; i < tensor_name.size(); ++i) {
			get_tensor(tensor_name[i], m_name, "Conv2D", m_weights[i]);
		}
		// check consistency of weight tensor's size
		for (size_t i = 0; i < tensor_name.size(); ++i) {
			if (m_weights[i]->size() != tensor_size[i]) {
				LOG(ERROR) << "Size mismatch of tensor [" << m_weights[i]->name()
					<< "] between weight file and model file (" << m_weights[i]->size()
					<< ", " << tensor_size[i] << ")!" << std::endl;
				return false;
			}
		}
		// set weight tensor's shape
		for (size_t i = 0; i < tensor_name.size(); ++i) {
			m_weights[i]->set_shape(tensor_shape[i]);
		}
		return true;
	}

	bool Conv2D::reshape(const std::vector<Tensor *> &input,
		std::vector<Tensor *> &output) {
		if (0 != input[0]->size() % m_input_size) {
			LOG(ERROR) << "Input size [" << input[0]->size() << "] should be divided by ["
				<< m_input_size << "]!" << std::endl;
			return false;
		}
		std::cout << input[0]->num_axes() ;
		std::cout << "\n";
		if (3 != input[0]->num_axes()) {
			LOG(ERROR) << "Only support 2d input of shape C*W*H!" << std::endl;
			return false;
		}
		std::vector<size_t> shape; //按照channel_first

		//计算输出的特征图
		const int out_w = (input[0]->shape(1) + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) / m_stride + 1;
		const int out_h = (input[0]->shape(2) + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) / m_stride + 1;

		shape.push_back(m_output_size);
		if (b_bias) {
			m_bias_multiplier.realloc(shape);
			lnn_set(shape[0], 1., m_bias_multiplier.data());
		}
		shape.push_back(out_w);
		shape.push_back(out_h);

		output[0]->realloc(shape);
		if (m_padding > 0) {
			shape = input[0]->shape();
			shape[0] += 2 * m_padding;
			m_buf.realloc(shape);
			lnn_set(m_padding*m_input_size, 0., m_buf.data());
			lnn_copy(input[0]->size(), input[0]->data(), m_buf.data() + m_padding * m_input_size);
			lnn_set(m_padding*m_input_size, 0., m_buf.data() + m_padding * m_input_size + input[0]->size());
		}
		if (m_dilation > 1) {
			shape.resize(1);
			shape[0] = m_kernel_size * m_input_size;
			m_buf_2.realloc(shape);
		}
		return true;
	}

	void Conv2D::forward_impl(const std::vector<Tensor *> &input,
		std::vector<Tensor *> &output) {
		size_t Map = input[0]->shape(1);
		size_t outM = (Map + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) / m_stride + 1;
		if (b_bias) {
			lnn_gemm(CblasNoTrans, CblasNoTrans, outM, m_output_size, 1,
				1., m_bias_multiplier.data(), m_weights[1]->data(), 0., output[0]->data());
		}
		else {
			lnn_set(output[0]->size(), 0., output[0]->data());
		}
		const float* data = input[0]->data();
		//1.将input进行padding
		if (m_padding > 0) data = m_buf.data();
		//2.将input进行convert(根据stride转换成对应的矩阵)
		//计算卷积输出矩阵宽高
		/*const int outM = Map - m_kernel_size + 1;*/

		//定义被卷积矩阵宽高
		const int convAw = m_kernel_size * m_kernel_size * m_input_size;
		const int convAh = outM * outM;

		float * a_convert = new float[convAh*convAw];
		float *A_convert = convert_x(data, a_convert, outM, m_input_size, Map, m_kernel_size);
		//3.cblas_gemm进行矩阵计算
		//for (size_t i = 0; i < convAh*convAw / 3; i++)
		//{
		//	for (size_t j = 0; j < 3; j++)
		//	{
		//		int k = i * 3 + j;
		//		std::cout << A_convert[k];
		//	}
		//	std::cout << std::endl;
		//}

		//定义cblas初始值
		//const int M = m_output_size;//A的行数，C的行数
		float* a = m_weights[0]->data();
		float* b = A_convert;

		//定义卷积输出矩阵
		//float C[M*N];
		//cblas计算输出矩阵 M=36 N=1 k=9。A_convert = [36,9]=36*9,B=[9,1]=9*1 C= [M,N]=M*N
		/*(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
			const int m, const int n, const int k, const float alpha,
			const float* a, const float* b, const float beta, float* c*/
		//cblas_sgemm(Order, TransA, TransB, m_output_size, N, K, alpha, m_weights[0]->data(), lda, A_convert, ldb, beta, output[0]->data(), ldc);
		lnn_gemm(CblasNoTrans, CblasTrans,m_output_size, convAh, convAw,1.0,a,b ,0.0, output[0]->data());

		delete[] a_convert;
		std::cout << sizeof(output[0]->data()) / sizeof(output[0]->data()[0]) << std::endl;
		//输出验证
		/*std::cout << "A is:" << std::endl;
		for (size_t k = 0; k < m_input_size; k++)
		{
			for (int i = 0; i < Map; i++)
			{
				for (int j = 0; j < Map; j++)
				{
					std::cout << data[k*Map*Map + i * Map + j] << " ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		std::cout << "B is:" << std::endl;
		for (size_t k = 0; k < m_input_size; k++)
		{
			for (size_t m = 0; m < m_output_size; m++)
			{
				for (int i = 0; i < m_kernel_size; i++)
				{
					for (int j = 0; j < m_kernel_size; j++)
					{
						std::cout << m_weights[0]->data()[k*m_output_size*m_kernel_size*m_kernel_size + m * m_kernel_size*m_kernel_size + i * m_kernel_size + j] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		std::cout << "C is:" << std::endl;
		for (size_t k = 0; k < m_output_size; k++)
		{
			for (int i = 0; i < outM; i++)
			{
				for (int j = 0; j < outM; j++)
				{
					std::cout << output[0]->data()[k*outM*outM + i * outM + j] << " ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;*/
	}
		
		/*if (m_dilation == 1) {
			for (size_t i = 0; i < O; ++i) {
				lnn_gemv(CblasNoTrans, m_output_size, m_kernel_size*m_input_size, 1.,
					m_weights[0]->data(), data + i * m_stride*m_input_size, 1.,
					output[0]->data() + i * m_output_size);
			}
		}
		else {
			for (size_t i = 0; i < O; ++i) {
				const float* cur = data + i * m_stride*m_input_size;
				for (size_t j = 0; j < m_kernel_size; ++j) {
					lnn_copy(m_input_size, cur + j * m_dilation*m_input_size, m_buf_2.data() + j * m_input_size);
				}
				lnn_gemv(CblasNoTrans, m_output_size, m_kernel_size*m_input_size, 1.,
					m_weights[0]->data(), m_buf_2.data(), 1.,
					output[0]->data() + i * m_output_size);
			}
		}*/
#ifdef DEBUG
		dump(output, std::cout);
#endif  // DEBUG
	}
   // namespace lnn
