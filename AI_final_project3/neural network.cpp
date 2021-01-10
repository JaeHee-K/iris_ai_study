#include <iostream>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <istream>
#include <stdlib.h>

using namespace std;

double Sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}
double D_Sigmoid(double x)
{
	double g = Sigmoid(x);
	return g*(1 - g);
}

int main()
{
	double input[4];
	const double alpha = 0.01; //�н��� , ���� �������� �� ��Ȯ��(�׷��� �ð��� ���̰ɸ�)
	double weight[3][4];//(�Է�->����)���ᰭ��
	double weight2[3][3]; //(����->���)���ᰭ��
	double out_change_weight[3][3]; //����ġ ��ȭ�� (���� ����ġ���� �÷��� �ؾ���)
	double hidden_change_weight[3][4]; // ������ ����ġ ��ȭ��
	double sig_result_input[3] = { 0 }; // �ñ׸��̵� �� ���
	double output_in[3] = { 0 };//��¿��� �޴� �Է�
	double output[3] = { 0 };//��°�
	double error[3] = { 0 };//������
	double delta[3] = { 0 };//(�����)��Ÿ��
	double hidden_delta[3] = { 0 };//(������)��Ÿ��
	double hidden_error[3] = { 0 };//(������)������
	double target_store[75][3] = { 0 };//Ÿ�ٰ� ����
	int count = 0;//Ʈ���̴� �Է°� ī��Ʈ
	int target_count = 0;//Ÿ�ٰ� ī��Ʈ(�ʱ�ȭ)
	int store_count = 0;//Ÿ�ٰ� ī��Ʈ(��������)
	double pure_input[3] = { 0 };//(������)�����Է�
	double output_pure_input[3] = { 0 };//(�����)�����Է�
	int t_count = 0;//�׽�Ʈ�� ī��Ʈ

	srand((unsigned int)time(NULL));

	///////////���� �����//////////////////
	vector<string> row, target, test;
	ifstream training_file("C://Users/KIMJAEHEE/Desktop/training_input2.csv");
	string str,str1,str2;
	while (getline(training_file, str, ','))
	{
		row.push_back(str);
	}

	ifstream target_file("C://Users/KIMJAEHEE/Desktop/training_target2.csv");
	while (getline(target_file, str1, ','))
	{
		target.push_back(str1);
	}

	ifstream test_file("C://Users/KIMJAEHEE/Desktop/testing2.csv");
	while (getline(test_file, str2, ','))
	{
		test.push_back(str2);
	}
	/////////////////////////////////////////

	for (int i = 0; i < 3; i++) // ���ᰭ��1(�Է�->����) ���������� �ʱ�ȭ
	{
		for (int j = 0; j < 4; j++)
		{
			weight[i][j] = ((double)rand() / RAND_MAX) - 0.5;
		}
	}
	
	for (int i = 0; i < 3; i++) // ���ᰭ��2(����->���) ���������� �ʱ�ȭ
	{
		for (int j = 0; j < 3; j++)
		{
			weight2[i][j] = ((double)rand() / RAND_MAX) - 0.5;
		}
	}

	for (int i = 0; i < 75; i++)//Ÿ�ٰ� ����
	{
		for (int j = 0; j < 3; j++)
		{
			target_store[i][j] = stod(target[target_count]);
			target_count++;
		}
	}


	for (int learning = 0; learning < 100; learning++)
	{
		for (int class1 = 0; class1 < 75; class1++)
		{
			for (int input_count = 0; input_count < 4; input_count++) // �Է°� 4���� ��� �ֱ�
			{
				if (count == 300)
				{
					count = 0;
				}
				input[input_count] = stod(row[count]);
				
				count++;
			}

			double sum = 0;
			int k = 0;
			while (k < 3)
			{
				//����� �۾��� ���� �����Է� ���ϱ�
				for (int i = 0; i < 4; i++)
				{
					sum += input[i] * weight[k][i];
				}
				sig_result_input[k] = Sigmoid(sum); // ���������� �����Է��� ������ �ñ׸��̵带 ���
				pure_input[k] = sum; // ������ �����Է�

				sum = 0;
				k++;
			}

			int k1 = 0;
			while (k1 < 3)
			{
				//����� �۾��� ���� ��°��
				for (int i = 0; i < 3; i++)
				{
					sum += sig_result_input[i] * weight2[k1][i];
				}
				output[k1] = Sigmoid(sum);// ���� ���
				output_pure_input[k1] = sum; // ����� �����Է�

				sum = 0;
				k1++;
			}

			for (int i = 0; i < 3; i++)//�������ϱ�
			{
				error[i] = target_store[store_count][i] - output[i];
			}
			store_count++;

			if (store_count == 75)
			{
				store_count = 0;
			}

			for (int i = 0; i < 3; i++) //��Ÿ ���ϱ�
			{
				delta[i] = error[i] * D_Sigmoid(output_pure_input[i]);
			}

			for (int i = 0; i < 3; i++) // (����<->���)����ġ ��ȭ�� ���ϱ�
			{
				for (int j = 0; j < 3; j++)
				{
					out_change_weight[i][j] = alpha * delta[i] * sig_result_input[j];
				}
			}

			for (int i = 0; i < 3; i++) // (����<->���)����ġ ����
			{
				for (int j = 0; j < 3; j++)
				{
					weight2[i][j] = out_change_weight[i][j] + weight2[i][j];
				}
			}

			int k2 = 0;
			while (k2 < 3)
			{
				//������ ��Ÿ�� ���� ���� ���ϱ�

				for (int i = 0; i < 3; i++)
				{
					sum += delta[i] * weight2[i][k2];
				}
				hidden_error[k2] = sum;
				sum = 0;
				k2++;
			}

			for (int i = 0; i < 3; i++) //(������)��Ÿ ���ϱ�
			{
				hidden_delta[i] = hidden_error[i] * D_Sigmoid(pure_input[i]);
			}

			for (int i = 0; i < 3; i++) // (�Է�<->����)����ġ ��ȭ�� ���ϱ�
			{
				for (int j = 0; j < 4; j++)
				{
					hidden_change_weight[i][j] = alpha * hidden_delta[i] * input[j];
				}
			}

			for (int i = 0; i < 3; i++) // (�Է�<->����)����ġ ����
			{
				for (int j = 0; j < 4; j++)
				{
					weight[i][j] = hidden_change_weight[i][j] + weight[i][j];
				}
			}

			/*for (int s = 0; s < 3; s++)
			{
				cout << output[s] << endl;
			}*/
		}
	}
	
	
	//�׽�Ʈ�� �ֱ�
	for (int class1 = 0; class1 < 75; class1++)
	{
		for (int input_count = 0; input_count < 4; input_count++) // �Է°� 4���� ��� �ֱ�
		{
			input[input_count] = stod(test[t_count]);
			t_count++;
		}

		double sum = 0;
		int k = 0;
		while (k < 3)
		{
			//����� �۾��� ���� �����Է� ���ϱ�
			for (int i = 0; i < 4; i++)
			{
				sum += input[i] * weight[k][i];
			}
			sig_result_input[k] = Sigmoid(sum); // ���������� �����Է��� ������ �ñ׸��̵带 ���

			sum = 0;
			k++;
		}

		int k1 = 0;
		while (k1 < 3)
		{
			//����� �۾��� ���� ��°��
			for (int i = 0; i < 3; i++)
			{
				sum += sig_result_input[i] * weight2[k1][i];
			}
			output[k1] = Sigmoid(sum);// ���� ���

			sum = 0;
			k1++;
		}	

		cout << "-----�������---" << class1 << endl;
		for (int s = 0; s < 3; s++)
		{
			cout << output[s] << endl;
		}
	}
	return 0;
}
