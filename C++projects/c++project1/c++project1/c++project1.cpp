//cblas加速二维矩阵卷积操作
//注意：默认pad=0，stride=1
//作者：samylee

#include <cblas.h>  
#include <iostream>

using namespace std;

int main()
{
	//定义被卷积矩阵
	const int Map = 8;
	const float A[Map * Map] = {
		1,2,3,4,5,6,7,8,
		1,2,3,4,5,6,7,8,
		1,2,3,4,5,6,7,8,
		1,2,3,4,5,6,7,8,
		1,2,3,4,5,6,7,8,
		1,2,3,4,5,6,7,8,
		1,2,3,4,5,6,7,8,
		1,2,3,4,5,6,7,8 };

	//定义卷积核
	const int Kernel = 3;
	const float B[Kernel * Kernel] = {
		1,1,1,
		1,1,1,
		1,1,1 };

	//计算卷积输出矩阵宽高
	const int outM = Map - Kernel + 1;

	//定义被卷积矩阵宽高
	const int convAw = Kernel * Kernel;
	const int convAh = outM * outM;

	//转换被卷积矩阵
	float A_convert[convAh*convAw] = { 0 };
	for (int i = 0; i < outM; i++)
	{
		for (int j = 0; j < outM; j++)
		{
			int wh = i * outM * convAw + j * convAw;

			int col1 = i * Map + j;
			A_convert[wh] = A[col1];
			A_convert[wh + 1] = A[col1 + 1];
			A_convert[wh + 2] = A[col1 + 2];

			int col2 = (i + 1) * Map + j;
			A_convert[wh + 3] = A[col2];
			A_convert[wh + 4] = A[col2 + 1];
			A_convert[wh + 5] = A[col2 + 2];

			int col3 = (i + 2) * Map + j;
			A_convert[wh + 6] = A[col3];
			A_convert[wh + 7] = A[col3 + 1];
			A_convert[wh + 8] = A[col3 + 2];
		}
	}

	//定义cblas初始值
	const enum CBLAS_ORDER Order = CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
	const int M = convAh;//A的行数，C的行数
	const int N = 1;//B的列数，C的列数
	const int K = convAw;//A的列数，B的行数
	const float alpha = 1;
	const float beta = 0;
	const int lda = K;//A的列
	const int ldb = N;//B的列
	const int ldc = N;//C的列

	//定义卷积输出矩阵
	float C[M*N];
	//cblas计算输出矩阵
	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A_convert, lda, B, ldb, beta, C, ldc);

	//输出验证
	cout << "A is:" << endl;
	for (int i = 0; i < Map; i++)
	{
		for (int j = 0; j < Map; j++)
		{
			cout << A[i*Map + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "B is:" << endl;
	for (int i = 0; i < Kernel; i++)
	{
		for (int j = 0; j < Kernel; j++)
		{
			cout << B[i*Kernel + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "C is:" << endl;
	for (int i = 0; i < outM; i++)
	{
		for (int j = 0; j < outM; j++)
		{
			cout << C[i*outM + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	system("pause");
	return EXIT_SUCCESS;
}