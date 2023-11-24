#include <vector>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <climits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <string>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <random>
#include <queue>
#include <deque>
#include <list>
#include <map>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_map>

using namespace std;

#define INPUT 18
#define H_PARAMS 5
#define OUTPUT 1
#define LAYER 3
#define TRAIN 100000//100000
#define TEST 1000
#define dr 0.01//学習率(0.01が良好)
    
double Bi[LAYER];
double Wi[LAYER][H_PARAMS][H_PARAMS];
double W0[H_PARAMS][INPUT];
int data_field[1000000][INPUT];
int data_pl[1000000];
int NOW;

double evaluate(double Y[OUTPUT]){//他クラス値分類
    
    return (Y[0]-data_pl[NOW])*(Y[0]-data_pl[NOW])/2.0;
}

double activ(double x){//活性化関数
    return max(0.0,x);
}
int rnd(int mini, int maxi) {
	static mt19937 mt((int)time(0));
	uniform_int_distribution<int> dice(mini, maxi);
	return dice(mt);
}

double d_rnd() {
	static mt19937 mt((int)time(0));
	uniform_real_distribution<double> dice(0, 1.0);
	return dice(mt);
}

void LAYER1(double X[INPUT],double Y[H_PARAMS]){

        for(int j=0;j<H_PARAMS;j++){
        for(int i=0;i<INPUT;i++){
        if(i==0){Y[j]=Bi[0];}
        Y[j]+=X[i]*W0[j][i];
        }
        Y[j]=activ(Y[j]);
        }
    
}

void LAYER2(double X[H_PARAMS],double Y[H_PARAMS],int type){
       
        for(int j=0;j<H_PARAMS;j++){
        for(int i=0;i<H_PARAMS;i++){
        if(i==0){Y[j]=Bi[type];}
        Y[j]+=X[i]*Wi[type][j][i];
        }
        Y[j]=activ(Y[j]);
        }
    
}

void LAYER3(double X[H_PARAMS],double Y[OUTPUT]){
      
        for(int j=0;j<OUTPUT;j++){
        for(int i=0;i<H_PARAMS;i++){ 
        if(i==0){Y[j]=Bi[LAYER-1];}
        Y[j]+=X[i]*Wi[LAYER-1][j][i];
        }
        Y[j]=activ(Y[j]);
        }
    
}

double loss(double X[INPUT]){

    double X2[LAYER-1][H_PARAMS];
    double Y[OUTPUT];
    
    LAYER1(X,X2[0]);
    for(int i=0;i<=LAYER-3;i++){
    LAYER2(X2[i],X2[i+1],i+1);
    }
    LAYER3(X2[LAYER-2],Y);
    
    return evaluate(Y);
}
double predict(double X[INPUT]){
    double X2[LAYER-1][H_PARAMS];
    double Y[OUTPUT];
    
    LAYER1(X,X2[0]);
    for(int i=0;i<=LAYER-3;i++){
    LAYER2(X2[i],X2[i+1],i+1);
    }
    LAYER3(X2[LAYER-2],Y);
    
    return Y[0];
}
double judge(int board[3][3]){
    
    int alpha=0;
    
    for(int y=0;y<3;y++){
    int sum[3]={0};
    for(int x=0;x<3;x++){
    sum[board[y][x]]++;
    }
    alpha=max(alpha,sum[1]);
    }
    
    double reach[4];
    reach[0]=alpha;
    
    alpha=0;
    
    for(int x=0;x<3;x++){
    int sum[3]={0};
    for(int y=0;y<3;y++){
    sum[board[y][x]]++;
    }
    alpha=max(alpha,sum[1]);
    }
    
    reach[1]=alpha;
    alpha=0;
    
    int s[3]={0};
    
    s[board[0][0]]++;
    s[board[1][1]]++;
    s[board[2][2]]++;
    
    if(alpha<s[1]){alpha=s[1];} 
    
    reach[2]=alpha;
    
    alpha=0;
    
    int s2[3]={0};
    
    s2[board[0][2]]++;
    s2[board[1][1]]++;
    s2[board[2][0]]++;
    
    if(alpha<s2[1]){alpha=s2[1];}
    
    reach[3]=alpha;
    
    return (double)(reach[3]+4*reach[2]+16*reach[1]+64*reach[0]);
}

void train(){

double X[INPUT];
    
double temp_Bi[LAYER];

double temp_Wi[LAYER][H_PARAMS][H_PARAMS];

double temp_W0[H_PARAMS][INPUT];    
    
double minl;
    

while(1){
    
    //NOW=0;
    int board[3][3];
    for(int j=0;j<9;j++){
        int r=rnd(0,2);
        //data_field[NOW][j]=r;
        board[j/3][j%3]=r;
    }
    int tugi[INPUT]={0};
    for(int j=0;j<9;j++){
        if(board[j/3][j%3]==1){
        tugi[j]=1;
        }
        else if(board[j/3][j%3]==2){
        tugi[9+j]=1;
        }
    }
    for(int j=0;j<INPUT;j++){
    data_field[NOW][j]=tugi[j];
    }
    data_pl[NOW]=judge(board);
    
for(int i=0;i<INPUT;i++){
X[i]=(double)data_field[NOW][i];
}

minl=loss(X);
    
memcpy(temp_Bi,Bi,sizeof(temp_Bi));
memcpy(temp_Wi,Wi,sizeof(temp_Wi));
memcpy(temp_W0,W0,sizeof(temp_W0));    

for(int i=0;i<LAYER;i++){
double r=d_rnd()*dr;
if(d_rnd()<=0.5){Bi[i]=Bi[i]+r;}
else{Bi[i]=Bi[i]-r;}
}

for(int i=0;i<H_PARAMS;i++){
for(int j=0;j<INPUT;j++){
double r=d_rnd()*dr;
if(d_rnd()<=0.5){W0[i][j]+=r;}
else{W0[i][j]-=r;}
}
}    

for(int i=0;i<LAYER;i++){
for(int j=0;j<H_PARAMS;j++){
for(int k=0;k<H_PARAMS;k++){
double r=d_rnd()*dr;
if(d_rnd()<=0.5){Wi[i][j][k]=Wi[i][j][k]+r;}
else{Wi[i][j][k]=Wi[i][j][k]-r;}
}
}
}
    
double l=loss(X);    
    
if(minl>l){
minl=l;
break;
}
else{
memcpy(Bi,temp_Bi,sizeof(temp_Bi));
memcpy(Wi,temp_Wi,sizeof(temp_Wi));
memcpy(W0,temp_W0,sizeof(temp_W0));    
}
    

}
    
}
    
double test(){
    
    /*
    
    FILE *fp;
    
    fp = fopen("puzz.txt", "r");    
    
    for(int i=0;i<TRAIN;i++){
    char str[INPUT];
    int pal;
    fscanf(fp, "%[^,],%d",str, &pal);
    for(int j=0;j<INPUT;j++){
    data_field[i][j]=stoii(str[j]);
    }
    data_pl[i]=pal;
    }
    
    fclose(fp);
    
    
    for(int i=0;i<TRAIN;i++){
    printf("train=%d/%d\n",i+1,TRAIN);
    NOW=i;
    train();
    }
    
    fp = fopen("test.txt", "r");    
    
    for(int i=0;i<TEST;i++){
    char str[INPUT];
    int pal;
    fscanf(fp, "%[^,],%d",str, &pal);
    for(int j=0;j<INPUT;j++){
    data_field[i][j]=stoii(str[j]);
    }
    data_pl[i]=pal;
    }
    
    fclose(fp);
    
    double X[INPUT];
    
    double sum=0;
    
    
    for(int t=0;t<TEST;t++){
    NOW=t;
    for(int i=0;i<INPUT;i++){
        X[i]=(double)data_field[t][i];
    }
    sum+=(double)loss(X);
    printf("true=%d,predict=%d\n",data_pl[t],predict(X));        
    }
    
   return sum/(double)TEST;
    */
    
    /*
    
    for(int i=0;i<TRAIN;i++){
    int board[3][3];
    for(int j=0;j<INPUT;j++){
        int r=rnd(0,2);
        data_field[i][j]=r;
        board[j/3][j%3]=r;
    }
    data_pl[i]=judge(board);   
    }
    */

    for(int i=0;i<LAYER;i++){
        double r=d_rnd();
        Bi[i]=r;
    }
    for(int i=0;i<H_PARAMS;i++){
    for(int j=0;j<INPUT;j++){
        double r=d_rnd();
        W0[i][j]=r;
    }
    }  
    for(int i=0;i<LAYER;i++){
    for(int j=0;j<H_PARAMS;j++){
    for(int k=0;k<H_PARAMS;k++){
    double r=d_rnd();
    Wi[i][j][k]=r;
    }
    }
    }
    
    for(int i=0;i<TRAIN;i++){
    if(i%10==0){
    printf("train=%d/%d\n",i,TRAIN);
    }
    NOW=i;
    train();
    }
    
    double X[INPUT];
    
    double avg=0;
    
    int counter=0;
    
    for(int t=0;t<TEST;t++){
    NOW=t;
    int board[3][3];
    for(int i=0;i<9;i++){
    int r=rnd(0,2);
    //data_field[t][i]=r;
    board[i/3][i%3]=r;
    //X[i]=(double)r;
    }
    int tugi[INPUT]={0};
    for(int j=0;j<9;j++){
        if(board[j/3][j%3]==1){
        tugi[j]=1;
        }
        else if(board[j/3][j%3]==2){
        tugi[9+j]=1;
        }
    }
    for(int j=0;j<INPUT;j++){
    data_field[t][j]=tugi[j];
    X[j]=(double)tugi[j];
    }
    data_pl[t]=judge(board);
    
    double truth=data_pl[t];
        
    double pred=predict(X);   
    //sum+=(double)loss(X);
    //for(int i=0;i<(int)truth.size();i++){
    printf("truth=%.1f,pred=%.1f\n",truth,pred);      
    //}  
        
    avg+=fabs(truth-pred);    
        
    }
    
    return avg/(double)TEST;
    
}


int main(){

printf("loss=%lf\n",test());

return 0;
}
