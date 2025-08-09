void Conv2D_0(float Input_Conv[784],float Output_Conv[21632], float bias[32], float kernel[288]){
	int stride = 1;
	loop_for_channel2D_0:
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_0:
		for (int x = 0; x < 26; x++){
			loop_for_ap2D_0:
			for (int y = 0; y < 26; y++){
				float s = 0;
				loop_for_fc_0:
				for (int k = 0; k < 1; k++){
					loop_for_fb_0:
					for (int i = 0; i < 3; i++){
						loop_for_fa_0:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[1*3*3*n+3*3*k+3*i+j])*(Input_Conv[28*28*k+28*(i+x*stride)+j+y*stride]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[26*26*n+26*x+y]=0; else Output_Conv[26*26*n+26*x+y]=s+bias[n];
			}
		}
	}
}
void Conv2D_1(float Input_Conv[5408],float Output_Conv[7744], float bias[64], float kernel[18432]){
	int stride = 1;
	loop_for_channel2D_1:
	for (int n = 0; n < 64; n++){
		loop_for_bp2D_1:
		for (int x = 0; x < 11; x++){
			loop_for_ap2D_1:
			for (int y = 0; y < 11; y++){
				float s = 0;
				loop_for_fc_1:
				for (int k = 0; k < 32; k++){
					loop_for_fb_1:
					for (int i = 0; i < 3; i++){
						loop_for_fa_1:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[13*13*k+13*(i+x*stride)+j+y*stride]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[11*11*n+11*x+y]=0; else Output_Conv[11*11*n+11*x+y]=s+bias[n];
			}
		}
	}
}
