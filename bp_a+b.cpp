#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define Data  820       // Data 用来表示已经知道的数据样本的数量，也就是训练样本的数量。
#define In 2            //In 表示对于每个样本有多少个输入变量
#define Out 1           // Out 表示对于每个样本有多少个输出变量
#define Neuron 45       //Neuron 表示神经元的数量
#define TrainC 20000    //TrainC 来表示训练的次数
#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3

double d_in[Data][In],d_out[Data][Out];
double w[Neuron][In],o[Neuron],v[Out][Neuron];
double Maxin[In],Minin[In],Maxout[Out],Minout[Out];
double OutputData[Out];
double dv[Out][Neuron],dw[Neuron][In];
double e;


void writeTest(){
	FILE *fp1,*fp2;
	double r1,r2;
	int i;
	srand((unsigned)time(NULL));
	if((fp1=fopen("C:\\Users\\liush\\Downloads\\neuron\\in.txt","w"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	if((fp2=fopen("C:\\Users\\liush\\Downloads\\neuron\\out.txt","w"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}


	for(i=0;i<Data;i++){
		r1=rand()%1000/100.0;
		r2=rand()%1000/100.0;
		fprintf(fp1,"%lf  %lf\n",r1,r2);
		fprintf(fp2,"%lf \n",r1+r2);
	}
	fclose(fp1);
	fclose(fp2);
}

void readData(){

	FILE *fp1,*fp2;
	int i,j;
	if((fp1=fopen("C:\\Users\\liush\\Downloads\\neuron\\in.txt","r"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	for(i=0;i<Data;i++)
		for(j=0; j<In; j++)
			fscanf(fp1,"%lf",&d_in[i][j]);
	fclose(fp1);

	if((fp2=fopen("C:\\Users\\liush\\Downloads\\neuron\\out.txt","r"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}
	for(i=0;i<Data;i++)
		for(j=0; j<Out; j++)
			fscanf(fp1,"%lf",&d_out[i][j]);
	fclose(fp2);
}

//初始化BP神经网络
void initBPNework(){

	int i,j;

	/*
        找到数据最小、最大值
    */
	for(i=0; i<In; i++){
		Minin[i]=Maxin[i]=d_in[0][i];   //初始化
		for(j=0; j<Data; j++)
		{
			Maxin[i]=Maxin[i]>d_in[j][i]?Maxin[i]:d_in[j][i];   //找输入的最大值
			Minin[i]=Minin[i]<d_in[j][i]?Minin[i]:d_in[j][i];   //找输入的最小值
		}
	}

	for(i=0; i<Out; i++){
		Minout[i]=Maxout[i]=d_out[0][i];    //初始化
		for(j=0; j<Data; j++)
		{
			Maxout[i]=Maxout[i]>d_out[j][i]?Maxout[i]:d_out[j][i];  //找输出的最大值
			Minout[i]=Minout[i]<d_out[j][i]?Minout[i]:d_out[j][i];  //找输出的最小值
		}
	}

    /*
　　　　归一化处理：将数据转换成0~1之间
        实际实践过程中，归一化处理是不可或缺的。因为理论模型没考虑到，BP神经网络收敛的速率问题，
        一般来说神经元的输出对于0~1之间的数据非常敏感，
        归一化能够显著提高训练效率。可以用以下公式来对其进行归一化，
        其中 加个常数A 是为了防止出现 0 的情况（0不能为分母）。
    */
	for (i = 0; i < In; i++)
		for(j = 0; j < Data; j++)
			d_in[j][i]=(d_in[j][i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1);

	for (i = 0; i < Out; i++)
		for(j = 0; j < Data; j++)
			d_out[j][i]=(d_out[j][i]-Minout[i]+1)/(Maxout[i]-Minout[i]+1);

    /*
　　　　初始化神经元
    */
	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < In; ++j){
			w[i][j]=rand()*2.0/RAND_MAX-1;
			dw[i][j]=0;
		}

    for (i = 0; i < Neuron; ++i)
        for (j = 0; j < Out; ++j){
            v[j][i]=rand()*2.0/RAND_MAX-1;
            dv[j][i]=0;
        }
}

//函数 computO(i) 负责的是通过BP神经网络的机制对样本 i 的输入，预测其输出
void computO(int var){

	int i,j;
	double sum,y;

    /*
        神经元输出
    */
	for (i = 0; i < Neuron; ++i){
		sum=0;
		for (j = 0; j < In; ++j)
			sum+=w[i][j]*d_in[var][j];
		o[i]=1/(1+exp(-1*sum));
	}

    /*  隐藏层到输出层输出 */
	for (i = 0; i < Out; ++i){
		sum=0;
		for (j = 0; j < Neuron; ++j)
			sum+=v[i][j]*o[j];

		OutputData[i]=sum;
	}
}

//函数 backUpdate(i) 负责的是将预测输出的结果与样本真实的结果进行比对，
//然后对神经网络中涉及到的权重进行修正，也这是BP神经网络实现的关键所在
void backUpdate(int var)
{
	int i,j;
	double t;
	for (i = 0; i < Neuron; ++i)
	{
		t=0;
		for (j = 0; j < Out; ++j){
			t+=(OutputData[j]-d_out[var][j])*v[j][i];

			dv[j][i]=A*dv[j][i]+B*(OutputData[j]-d_out[var][j])*o[i];
			v[j][i]-=dv[j][i];
		}

		for (j = 0; j < In; ++j){
			dw[i][j]=a*dw[i][j]+b*t*o[i]*(1-o[i])*d_in[var][j];
			w[i][j]-=dw[i][j];
		}
	}
}

double result(double var1,double var2)
{
	int i,j;
	double sum,y;

	var1=(var1-Minin[0]+1)/(Maxin[0]-Minin[0]+1);
	var2=(var2-Minin[1]+1)/(Maxin[1]-Minin[1]+1);

	for (i = 0; i < Neuron; ++i){
		sum=0;
		sum=w[i][0]*var1+w[i][1]*var2;
		o[i]=1/(1+exp(-1*sum));
	}
	sum=0;
	for (j = 0; j < Neuron; ++j)
		sum+=v[0][j]*o[j];

	return sum*(Maxout[0]-Minout[0]+1)+Minout[0]-1;
}

void writeNeuron()
{
	FILE *fp1;
	int i,j;
	if((fp1=fopen("C:\\Users\\liush\\Downloads\\neuron\\neuron.txt","w"))==NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < In; ++j){
			fprintf(fp1,"%lf ",w[i][j]);
		}
	fprintf(fp1,"\n\n\n\n");

	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < Out; ++j){
			fprintf(fp1,"%lf ",v[j][i]);
		}

	fclose(fp1);
}

/*
    由BP神经网络的基本模型知道，反馈学习机制包括两大部分，
    一是BP神经网络产生预测的结果，
    二是通过预测的结果和样本的准确结果进行比对，然后对神经元进行误差量的修正。
    因此，我们用两个函数来表示这样的两个过程，训练过程中还对平均误差 e 进行监控，
    如果达到了设定的精度即可完成训练。由于不一定能够到达预期设定的精度要求，
    我们添加一个训练次数的参数，如果次数达到也退出训练。
*/
void  trainNetwork(){

	int i,c=0,j;
	do{
		e=0;    //平均误差
		for (i = 0; i < Data; ++i){
			computO(i);
			for (j = 0; j < Out; ++j)
				e+=fabs((OutputData[j]-d_out[i][j])/d_out[i][j]);   //计算BP神经网络预测第 i 个样本的输出也就是第一个过程
			backUpdate(i);          //backUpdate(i) 是根据预测的第 i 个样本输出对神经网络的权重进行更新
		}
		printf("%d  %lf\n",c,e/Data);
		c++;
	}while(c<TrainC && e/Data>0.01);    //如果达到了设定的精度即可完成训练，如果次数达到也退出训练。
}



int  main(int argc, char const *argv[])
{
	writeTest();
	readData();
	initBPNework();
	trainNetwork();
	printf("%lf \n",result(6,8) );
	printf("%lf \n",result(2.1,7) );
	printf("%lf \n",result(4.3,8) );
	writeNeuron();
	return 0;
}
