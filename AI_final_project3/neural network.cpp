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
	const double alpha = 0.01; //학습률 , 값이 작을수록 더 정확함(그러나 시간이 많이걸림)
	double weight[3][4];//(입력->은닉)연결강도
	double weight2[3][3]; //(은닉->출력)연결강도
	double out_change_weight[3][3]; //가중치 변화값 (원래 가중치에서 플러스 해야함)
	double hidden_change_weight[3][4]; // 은닉층 가중치 변화값
	double sig_result_input[3] = { 0 }; // 시그모이드 후 결과
	double output_in[3] = { 0 };//출력에서 받는 입력
	double output[3] = { 0 };//출력값
	double error[3] = { 0 };//에러값
	double delta[3] = { 0 };//(출력층)델타값
	double hidden_delta[3] = { 0 };//(은닉층)델타값
	double hidden_error[3] = { 0 };//(은닉층)에러값
	double target_store[75][3] = { 0 };//타겟값 저장
	int count = 0;//트레이닝 입력값 카운트
	int target_count = 0;//타겟값 카운트(초기화)
	int store_count = 0;//타겟값 카운트(에러설정)
	double pure_input[3] = { 0 };//(은닉층)순수입력
	double output_pure_input[3] = { 0 };//(출력층)순수입력
	int t_count = 0;//테스트값 카운트

	srand((unsigned int)time(NULL));

	///////////파일 입출력//////////////////
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

	for (int i = 0; i < 3; i++) // 연결강도1(입력->은닉) 랜덤값으로 초기화
	{
		for (int j = 0; j < 4; j++)
		{
			weight[i][j] = ((double)rand() / RAND_MAX) - 0.5;
		}
	}
	
	for (int i = 0; i < 3; i++) // 연결강도2(은닉->출력) 랜덤값으로 초기화
	{
		for (int j = 0; j < 3; j++)
		{
			weight2[i][j] = ((double)rand() / RAND_MAX) - 0.5;
		}
	}

	for (int i = 0; i < 75; i++)//타겟값 저장
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
			for (int input_count = 0; input_count < 4; input_count++) // 입력값 4개에 요소 넣기
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
				//가산기 작업을 통해 순수입력 구하기
				for (int i = 0; i < 4; i++)
				{
					sum += input[i] * weight[k][i];
				}
				sig_result_input[k] = Sigmoid(sum); // 은식층에서 순수입력을 가지고 시그모이드를 사용
				pure_input[k] = sum; // 은닉층 순수입력

				sum = 0;
				k++;
			}

			int k1 = 0;
			while (k1 < 3)
			{
				//가산기 작업을 통해 출력계산
				for (int i = 0; i < 3; i++)
				{
					sum += sig_result_input[i] * weight2[k1][i];
				}
				output[k1] = Sigmoid(sum);// 실제 출력
				output_pure_input[k1] = sum; // 출력층 순수입력

				sum = 0;
				k1++;
			}

			for (int i = 0; i < 3; i++)//에러구하기
			{
				error[i] = target_store[store_count][i] - output[i];
			}
			store_count++;

			if (store_count == 75)
			{
				store_count = 0;
			}

			for (int i = 0; i < 3; i++) //델타 구하기
			{
				delta[i] = error[i] * D_Sigmoid(output_pure_input[i]);
			}

			for (int i = 0; i < 3; i++) // (은닉<->출력)가중치 변화값 구하기
			{
				for (int j = 0; j < 3; j++)
				{
					out_change_weight[i][j] = alpha * delta[i] * sig_result_input[j];
				}
			}

			for (int i = 0; i < 3; i++) // (은닉<->출력)가중치 조정
			{
				for (int j = 0; j < 3; j++)
				{
					weight2[i][j] = out_change_weight[i][j] + weight2[i][j];
				}
			}

			int k2 = 0;
			while (k2 < 3)
			{
				//은닉층 델타를 위한 에러 구하기

				for (int i = 0; i < 3; i++)
				{
					sum += delta[i] * weight2[i][k2];
				}
				hidden_error[k2] = sum;
				sum = 0;
				k2++;
			}

			for (int i = 0; i < 3; i++) //(은닉층)델타 구하기
			{
				hidden_delta[i] = hidden_error[i] * D_Sigmoid(pure_input[i]);
			}

			for (int i = 0; i < 3; i++) // (입력<->은닉)가중치 변화값 구하기
			{
				for (int j = 0; j < 4; j++)
				{
					hidden_change_weight[i][j] = alpha * hidden_delta[i] * input[j];
				}
			}

			for (int i = 0; i < 3; i++) // (입력<->은닉)가중치 조정
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
	
	
	//테스트값 넣기
	for (int class1 = 0; class1 < 75; class1++)
	{
		for (int input_count = 0; input_count < 4; input_count++) // 입력값 4개에 요소 넣기
		{
			input[input_count] = stod(test[t_count]);
			t_count++;
		}

		double sum = 0;
		int k = 0;
		while (k < 3)
		{
			//가산기 작업을 통해 순수입력 구하기
			for (int i = 0; i < 4; i++)
			{
				sum += input[i] * weight[k][i];
			}
			sig_result_input[k] = Sigmoid(sum); // 은식층에서 순수입력을 가지고 시그모이드를 사용

			sum = 0;
			k++;
		}

		int k1 = 0;
		while (k1 < 3)
		{
			//가산기 작업을 통해 출력계산
			for (int i = 0; i < 3; i++)
			{
				sum += sig_result_input[i] * weight2[k1][i];
			}
			output[k1] = Sigmoid(sum);// 실제 출력

			sum = 0;
			k1++;
		}	

		cout << "-----실제출력---" << class1 << endl;
		for (int s = 0; s < 3; s++)
		{
			cout << output[s] << endl;
		}
	}
	return 0;
}
