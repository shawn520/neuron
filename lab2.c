#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(){

    	
	double a[1000][2];
	double b[1000];

	//数组a、b即可作为网络的训练样本。

	//随机初始化1000对数值在0~10之间的浮点数,保存在二维数组a[1000][2]中。
	for(int i=0;i<1000;i++){
		a[i][0] = rand() /(double)(RAND_MAX/10);
		//printf("%f\t",a[i][0]);

		a[i][1] = rand() /(double)(RAND_MAX/10);
		//printf("%f\t",a[i][1]);

		//计算各对浮点数的相加结果,保存在数组b[1000]中,即b[0] = a[0][0] + a[0][1],以此类推。
		b[i] = a[i][0] + a[i][1];
	}
	return 0;
}

