#include <cmath>
void Dense_0(float input_Dense[1600],float &output_Dense1,float bias[10],float weight[16000]){
	float out_Dense[10];
	loop_for_a_Dense_0:
	for (int i = 0; i < 10; i++){
		float s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 1600; j++){
			s+=input_Dense[j]*weight[j*10+i];
		}
		out_Dense[i]=s+bias[i];
	}
	int maxindex = 0;
	float max=out_Dense[0];
	loop_detect:
	for (int i=0; i<10; i++){
		if (out_Dense[i]> max) {
			max=out_Dense[i];
			maxindex=i;
		}
	}
	float sum_exp_x = 0.0;
	for(int i = 0; i <10;i++){
		sum_exp_x += exp(out_Dense[i]- out_Dense[maxindex]);
	}
	float max_value = out_Dense[maxindex];
	for(int i = 0; i <10;i++){
		out_Dense[i] = exp(out_Dense[i] - max_value) / sum_exp_x;
	}
	float maxindex_2 = 0;
	float max_2 = out_Dense[0];
	for(int i = 0; i <10;i++){
		if (out_Dense[i] > max_2) {
			max_2 = out_Dense[i];
			maxindex_2 = i;
		}
	}
	output_Dense1 = maxindex_2;
}
