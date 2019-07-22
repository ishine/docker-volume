// c++project1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
using namespace std;

//有参的构造方法,类属性length，类方法set_length，get_length
class Line
{
public:
	void SetLength(double len);
	double GetLength();
	Line(double len);
	~Line();
private:
	double length;
};
//成员函数定义
void Line::SetLength(double len)
{
	length = len;
}

double Line::GetLength()
{
	return length;
}

//构造函数初始化
Line::Line(double len):length(len)
{
	cout << "Object is being created" << endl;
	//length = len;
}
Line::~Line()
{
	cout << "Object is being deleted" << endl;
}
int main() {
	Line line(10.0);
	cout << line.GetLength() << endl;
	line.SetLength(6.0);
	cout << line.GetLength() << endl;
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
