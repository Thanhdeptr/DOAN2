#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#include <algorithm>
#include <string.h>
void CNN(float InModel[784],float &OutModel0,float &OutModel1,float &OutModel2,float Weights[34826]){
	float conv2d[21632];
	float max_pooling2d[5408];
	float conv2d_1[7744];
	float max_pooling2d_1[1600];
	float flatten[1600];
	Conv2D_0(&InModel[0],conv2d,&Weights[288],&Weights[0]);
	Max_Pool2D_0(conv2d,max_pooling2d);
	Conv2D_1(max_pooling2d,conv2d_1,&Weights[18752],&Weights[320]);
	Max_Pool2D_1(conv2d_1,max_pooling2d_1);
	flatten0(max_pooling2d_1,flatten);
	Dense_0(flatten,OutModel0,&Weights[34816],&Weights[18816]);
}
