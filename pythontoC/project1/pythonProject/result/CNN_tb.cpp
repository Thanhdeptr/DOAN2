#define _CRT_SECURE_NO_WARNINGS
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include "CNN.h"
#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#define NumberOfPicture ...
#define d ...
int main(){
	float OutModel0, OutModel1, OutModel2;
	float* Weights = (float*)malloc(34826 * sizeof(float));
	float tmp;
	FILE* Weight = fopen("Float_Weights.txt", "r");
	for (int i = 0; i < 34826; i++){
		fscanf(Weight, "%f", &tmp);
		*(Weights + i)=tmp;
	}
	fclose(Weight);
	////read Input
	float* InModel = (float*)malloc((NumberOfPicture * d * 28 * 28) * sizeof(float));
	FILE* Input = fopen("X.txt", "r");
	for (int i = 0; i < NumberOfPicture * d * 28 * 28; i++){
		fscanf(Input, "%f", &tmp);
		*(InModel + i)=tmp;
	}
	fclose(Input);
	//Read Label
	float*Label = (float*)malloc((NumberOfPicture) * sizeof(float));
	FILE* Output = fopen("Y.txt", "r");
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		fscanf(Output, "%f", &tmp);
		*(Label + i) = tmp;
	}
	fclose(Output);
	float*OutArray = (float*)malloc((NumberOfPicture) * sizeof(float));
	float Image[d * 28 * 28] = {};
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		int startIndex = i * d * 28 * 28;
		for (int k = 0; k < d * 28 * 28; k++)
		{
			Image[k] = *(InModel + startIndex + k);
		}
		CNN(Image, OutModel0, OutModel1, OutModel2, Weights);
		*(OutArray + i) = OutModel0;
	}
	float countTrue = 0;
	for (int i = 0; i < NumberOfPicture; i++)
	{
		int labelValue = *(Label + i);
		int PredictValue = *(OutArray + i);
		if (labelValue == PredictValue)
		{
			countTrue = countTrue + 1;
		}
	}
	float accuracy = (float)((countTrue / (NumberOfPicture)) * 100);
	std::cout << "accuracy of Model: " << accuracy << "%\n";
	//std::cout << "Result: " <<  OutModel <<  "\n";
	return 0;
}
