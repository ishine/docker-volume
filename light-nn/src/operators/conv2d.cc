//头文件
#include "operators/conv2d.h"
#include "utils/math-functions.h"


namespace lnn {

	//float * convert_x(const float *A, float *A_convert, const int outM, const int in_channels, int Map, int Kernel)
	//{//转换被卷积矩阵 这里根据步长和kernel_size大小进行转换
	//	//定义被卷积矩阵宽高
	//	const int convAw = Kernel * Kernel * in_channels;
	//	const int convAh = outM * outM;
	//	//float A_convert[convh*convAw] = { 0 };
	//	for (int i = 0; i < outM; i++)
	//	{
	//		for (int j = 0; j < outM; j++)
	//		{
	//			int wh = i * outM * convAw + j * convAw;
	//			for (size_t k = 0; k < in_channels; k++)
	//			{
	//				//int wh = k * outM * outM *convAw+ i * outM * convAw + j * convAw;

	//				int col1 = k * Map * Map + i * Map + j; //一个k跨过map*map个元素，三维先拉伸成二维，再拉伸成一维
	//				for (size_t m = 0; m < Kernel; m++)
	//				{
	//					A_convert[wh+m] = A[col1+m];
	//				}

	//				int col2 = k * Map * Map + (i + 1) * Map + j;
	//				A_convert[wh + 3] = A[col2];
	//				A_convert[wh + 4] = A[col2 + 1];
	//				A_convert[wh + 5] = A[col2 + 2];

	//				int col3 = k * Map * Map + (i + 2) * Map + j;
	//				A_convert[wh + 6] = A[col3];
	//				A_convert[wh + 7] = A[col3 + 1];
	//				A_convert[wh + 8] = A[col3 + 2];
	//				wh += Kernel * Kernel;
	//			}
	//		}
	//	}

	//	return A_convert;
	//}
	float * convert_x(const float *A, float *A_convert, const int outM, const int in_channels, const int Map, const int Kernel, const int stride)
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

				//k是channels移动，i是行移动，j是列移动
				//stride体现在取值得过程中,取值的起始点
				int col = (i*stride*Map + j * stride);
				//将原图拉升成一维，填入到三维转换图中
				for (size_t k = 0; k < in_channels; k++)
				{
					col += (k * Map*Map);
					//适应不同大小的kernel_size
					int id = 0;
					for (size_t m = 0; m < Kernel; m++)
					{
						col += (m*Map);
						//col += Map;
						for (size_t n = 0; n < Kernel; n++)
						{
							A_convert[wh + id] = A[col + n];
							/*cout << "第" << i << "行";
							cout << "第" << j << "行";
							cout << "起始点下标" << col;
							cout << "第"<<(k+1)*(id+1)<<"个元素"<<A[col + n] << endl;*/
							id++;
						}
						//复位
						col -= (m*Map);
					}
					//复位
					col -= (k * Map*Map);

					wh += Kernel * Kernel;
				}
			}
		}

		return A_convert;
	}
	void padding_x(const float *A, float *new_A, const int padding, const int height, const int width, const int in_channels)
	{
		const int new_width = width + padding * 2;
		const int new_height = height + padding * 2;
		for (size_t c = 0; c < in_channels; c++)
		{
			size_t start = c * new_width*new_height;
			//遍历原数组中的元素添加到新数组中
			for (size_t i = 0; i < height; i++)
			{
				start += (i + padding) * new_width + padding;//每一个通道特征图的起始点
				for (size_t j = 0; j < width; j++)
				{
					new_A[start + j] = A[c*height*width + i*width + j];
				}
				//复位
				start -= (i + padding) * new_width + padding;
			}
		}
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
		shape.push_back(out_w);
		shape.push_back(out_h);
		output[0]->realloc(shape);
		if (m_padding > 0) {
			shape = input[0]->shape();
			const size_t new_w = input[0]->shape(1) + m_padding * 2;
			const size_t new_h = input[0]->shape(2) + m_padding * 2;
			shape[1] += 2 * m_padding;
			shape[2] += 2 * m_padding;
			m_buf.realloc(shape);
			lnn_set(shape[1] * shape[2] * m_input_size, 0., m_buf.data()); //m_buf初始化为0矩阵
			//float *new_A = new float[m_input_size*new_h*new_h]();

			padding_x(input[0]->data(), m_buf.data(), m_padding, input[0]->shape(1), input[0]->shape(2), m_input_size);

			//验证padding
			/*for (size_t i = 0; i < m_input_size; i++)
			{
				for (size_t j = 0; j < new_w; j++)
				{
					for (size_t k = 0; k < new_h; k++)
					{
						std::cout << m_buf.data()[i*new_w*new_h + j * new_h + k] << ",";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}*/
		}
		
		if (b_bias) {
			std::vector<size_t> t_shape;
			t_shape.push_back(out_w);
			t_shape.push_back(out_h);
			m_bias_multiplier.realloc(t_shape);
			lnn_set(t_shape[0]* t_shape[1], 1., m_bias_multiplier.data());
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
		//计算卷积输出矩阵宽高
		size_t outM = (Map + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) / m_stride + 1;
		if (b_bias) {
			lnn_gemm(CblasNoTrans, CblasNoTrans, m_output_size,outM*outM,  1,
				1., m_weights[1]->data(), m_bias_multiplier.data(), 0., output[0]->data());
		}
		else {
			lnn_set(output[0]->size(), 0., output[0]->data());
		}

	/*	std::cout << "矩阵计算后bias 的长度：" << output[0]->size() << "\t";
		for (size_t i = 0; i < output[0]->size(); i++)
		{
			std::cout << output[0]->data()[i] << "  ";
		}*/
		std::cout << std::endl;
		const float* data = input[0]->data();
		//1.将input进行padding
		if (m_padding > 0)
		{
			data = m_buf.data();
			Map = m_buf.shape(1);
		}
		//2.将input进行convert(根据stride转换成对应的矩阵)

		//验证padding
		LOG(INFO) << "conv2d padding" << std::endl;
		//for (size_t i = 0; i < m_input_size; i++)
		//{
		//	for (size_t j = 0; j < Map; j++)
		//	{
		//		for (size_t k = 0; k < Map; k++)
		//		{
		//			std::cout << data[i*Map*Map + j * Map + k] << ",";
		//		}
		//		std::cout << std::endl;
		//	}
		//	//break;
		//	std::cout << std::endl;
		//}
		//定义被卷积矩阵宽高
		const int convAw = m_kernel_size * m_kernel_size * m_input_size;
		const int convAh = outM * outM;

		float * a_convert = new float[convAh*convAw];
		float *A_convert = convert_x(data, a_convert, outM, m_input_size, Map, m_kernel_size, m_stride);
		//3.cblas_gemm进行矩阵计算
		LOG(INFO) << "conv2d convert matrix" << std::endl;
		//for (size_t i = 0; i < convAh*convAw / m_input_size; i++)
		//{
		//	for (size_t j = 0; j < m_input_size; j++)
		//	{
		//		int k = i * m_input_size + j;
		//		std::cout << A_convert[k];
		//	}
		//	/*if (i > m_kernel_size*m_kernel_size*m_input_size)
		//	{
		//		break;
		//	}*/
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
		lnn_gemm(CblasNoTrans, CblasTrans,m_output_size, convAh, convAw,1.0,a,b ,1.0, output[0]->data());

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
