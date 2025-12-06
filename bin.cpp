
/*

g++ -O2 -std=c++11 -fopenmp bin.cpp -o bin

*/
#pragma warning(disable:4710)
#pragma warning(disable:4711)
#pragma warning(disable:4820)
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
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
#define ROW 5
#define COL 6

int data2[10][10][ROW*COL][ROW*COL][ROW*COL][ROW*COL]={0};

int main() {


	int i, j, k;
	
	bool start_test=true;//koko
	if(start_test){
        ifstream myf ("data6.txt");
	    string ls;
	    while(getline(myf,ls)){
		    string parent="";
		    string child="";
		    bool slash=false;
		    for(i=0;i<(int)ls.size();i++){
			    if(ls[i]=='/'){slash=true;continue;}
			    if(slash){child+=ls[i];}
			    else{parent+=ls[i];}
		    }
		    int counter=0;
		    string xz[6]={"","","","","",""}; 
		    for(i=0;i<(int)parent.size();i++){
			    if(parent[i]==','){counter++;continue;}
			    xz[counter]+=parent[i];
		    }
		    data2[stoi(xz[0])][stoi(xz[1])][stoi(xz[2])][stoi(xz[3])][stoi(xz[4])][stoi(xz[5])]=stoi(child);
		    }
		myf.close();
	}
	int SIZ=100000;
	int pattern[SIZ]={0};
	int maxp=0;
	for (int a1=0;a1<10;a1++){
	for(int a2=0;a2<10;a2++){
	for(int a3=0;a3<ROW*COL;a3++){
	for(int a4=0;a4<ROW*COL;a4++){
	for(int a5=0;a5<ROW*COL;a5++){
	for(int a6=0;a6<ROW*COL;a6++){    
	int value=data2[a1][a2][a3][a4][a5][a6];    
 	pattern[value]++;
	if(value>maxp){maxp=value;}
 	}
	}
	}
	}
	}
	}
    int sum[5]={0};
    for(i=1;i<=10;i++){
    sum[0]+=pattern[i];
    }
    for(i=11;i<=50;i++){
    sum[1]+=pattern[i];
    }
    for(i=51;i<=500;i++){
    sum[2]+=pattern[i];
    }
    for(i=501;i<SIZ;i++){
    if(maxp<i){break;}
    sum[3]+=pattern[i];
    }
    printf("(0)=%d,(1-10)=%d,(11-50)=%d,(51-500)=%d,(501-%d)=%d\n",pattern[0],sum[0],sum[1],sum[2],maxp,sum[3]);
	return 0;
}
