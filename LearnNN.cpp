/*
実行コマンド:
g++ -O2 -std=c++11 -fopenmp LearnNN.cpp -o LearnNN
./LearnNN
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

// === パラメータ設定 ===
#define ROW 5
#define COL 6
#define DROP 8
#define TRN 150
#define MAX_TURN 150
#define BEAM_WIDTH 1000  // 学習用: 高速化のため1000に設定
#define BATCH 30         // 学習用: 安定化のためバッチサイズを増やす
#define PROBLEM 1000     // テスト用問題数
#define H_PARAMS1 128
#define H_PARAMS2 16
#define TEST 10000         // 学習の反復回数
#define BONUS 10

// === 重要な変更点: 配列サイズ拡張 ===
// ドロップ間距離(p2-p1)はマイナスになるため、オフセットを足して正の数として扱う
#define D_OFFSET (ROW*COL)
#define D_SIZE (2*ROW*COL) // サイズを倍確保する

#define NODE_SIZE MAX(500,4*BEAM_WIDTH)
#define DIR 4
#define DLT(ST,ED) ((double)((ED)-(ST))/CLOCKS_PER_SEC)
#define XX(PT)  ((PT)&15)
#define YY(PT)  XX((PT)>>4)
#define YX(Y,X) ((Y)<<4|(X))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef char F_T;
typedef char T_T;
typedef signed char sc;
typedef unsigned char uc;
typedef unsigned long long ll;

enum { EVAL_NONE = 0, EVAL_FALL, EVAL_SET, EVAL_FS, EVAL_COMBO };

// --- 関数プロトタイプ宣言 ---
void init(F_T field[ROW][COL]);
void fall(int x,int h,F_T field[ROW][COL]);
void set(F_T field[ROW][COL], int force);
void show_field(F_T field[ROW][COL]);
int rnd(int mini, int maxi);
int chain(int nrw, int ncl, F_T d, F_T field[ROW][COL], F_T chkflag[ROW][COL], F_T delflag[ROW][COL]);
int sum_e(F_T field[ROW][COL]);
int sum_evaluate(F_T field[ROW][COL]);
int sum_e2(F_T field[ROW][COL], sc* combo, ll* hash,int p_maxcombo[DROP+1]);
void operation(F_T field[ROW][COL], T_T first_te,ll route[(TRN/21)+1]);
ll xor128();

// --- グローバル変数 ---
ll zoblish_field[ROW][COL][DROP+1];

// ネットワーク構造体 (整数版: 推論用)
struct NNUE {
    int weights1[D_SIZE][D_SIZE][H_PARAMS1]; // サイズ拡張
    int weights2[H_PARAMS1][H_PARAMS2];
    int biases2[H_PARAMS2];
    int weights3[H_PARAMS2];
    int bias3;
} net;

// ネットワーク構造体 (実数版: 学習用)
struct NNUE_F {
    double weights1[D_SIZE][D_SIZE][H_PARAMS1]; // サイズ拡張
    double weights2[H_PARAMS1][H_PARAMS2];
    double biases2[H_PARAMS2];
    double weights3[H_PARAMS2];
    double bias3;
} net_f;

// 摂動構造体 (微小変化用)
struct NNUE_Delta {
    sc weights1[D_SIZE][D_SIZE][H_PARAMS1]; // サイズ拡張
    sc weights2[H_PARAMS1][H_PARAMS2];
    sc biases2[H_PARAMS2];
    sc weights3[H_PARAMS2];
    sc bias3;
} delta;

double max_avg=0;
bool change_weight=true;
int win=0;
int all_play=0;

// 探索ノード
struct node {
    T_T first_te;
    ll movei[(TRN/21)+1];
    int score;
    sc combo;
    sc nowC;
    sc nowR;
    sc prev;
    int prev_score;
    uc improving;
    ll hash;
    node() {
        this->score = 0;
        this->prev = -1;
    }
    bool operator < (const node& n)const {
        return score < n.score;
    }
} fff[NODE_SIZE];

struct Action {
    T_T first_te;
    int score;
    int maxcombo;
    ll moving[(TRN/21)+1];
    Action() {
        this->score = 0;
    }
};

Action BEAM_SEARCH(F_T f_field[ROW][COL]);
double part1 = 0, part2 = 0, part3 = 0, part4 = 0, MAXCOMBO = 0;

// --- 重み同期関数 ---
void sync_weights() {
    for (int i = 0; i < D_SIZE; i++)
        for (int j = 0; j < D_SIZE; j++)
            for (int k = 0; k < H_PARAMS1; k++)
                net.weights1[i][j][k] = (int)round(net_f.weights1[i][j][k]);

    for (int i = 0; i < H_PARAMS1; i++)
        for (int j = 0; j < H_PARAMS2; j++)
            net.weights2[i][j] = (int)round(net_f.weights2[i][j]);

    for (int i = 0; i < H_PARAMS2; i++) {
        net.biases2[i] = (int)round(net_f.biases2[i]);
        net.weights3[i] = (int)round(net_f.weights3[i]);
    }
    net.bias3 = (int)round(net_f.bias3);
}

// --- NNUEスコア計算 ---
int NNUE_score(F_T board[ROW][COL],int c1,int c2) {
    vector<int>v[10];
    for(int i=0;i<ROW*COL;i++){
        int a = (int)(board[i/COL][i%COL]);
        if(a==c1||a==c2){
        v[a].push_back(i);
        }
    }

    int input[H_PARAMS1] = {0};

    for(int i=0;i<10;i++){
    if(i==c1||i==c2){
        for(int j=0;j<(int)v[i].size();j+=3){
            for(int k=0;k<H_PARAMS1;k++){
                if((int)v[i].size()<=j+2){break;}
                int p1 = v[i][j];
                int p2 = v[i][j+1];
                int p3 = v[i][j+2];
               
                // 配列の添字がマイナスにならないようにオフセットを足す
                int d1 = (p2 - p1) + D_OFFSET;
                int d2 = (p3 - p1) + D_OFFSET;
               
                // 安全装置
                if(d1 >= 0 && d1 < D_SIZE && d2 >= 0 && d2 < D_SIZE) {
                    input[k] += net.weights1[d1][d2][k];
                }
            }
        }
    }
}
   
    // 中間層
    int hidden[H_PARAMS2] = {0};
    for (int i = 0; i < H_PARAMS2; i++) {
        for (int j = 0; j < H_PARAMS1; j++) {
            hidden[i] += input[j] * net.weights2[j][i];
        }
        hidden[i] += net.biases2[i];
        hidden[i] = max(0, hidden[i]); // ReLU
    }
   
    // 出力層
    int score = net.bias3;
    for (int i = 0; i < H_PARAMS2; i++) {
        score += hidden[i] * net.weights3[i];
    }
   
    return score;
}

int NNUE_init_score(F_T board[ROW][COL]) {
    vector<int>v[10];
    for(int i=0;i<ROW*COL;i++){
        int a = (int)(board[i/COL][i%COL]);
        v[a].push_back(i);
    }

    int input[H_PARAMS1] = {0};

    for(int i=0;i<10;i++){
        for(int j=0;j<(int)v[i].size();j+=3){
            for(int k=0;k<H_PARAMS1;k++){
                if((int)v[i].size()<=j+2){break;}
                int p1 = v[i][j];
                int p2 = v[i][j+1];
                int p3 = v[i][j+2];
               
                // 配列の添字がマイナスにならないようにオフセットを足す
                int d1 = (p2 - p1) + D_OFFSET;
                int d2 = (p3 - p1) + D_OFFSET;
               
                // 安全装置
                if(d1 >= 0 && d1 < D_SIZE && d2 >= 0 && d2 < D_SIZE) {
                    input[k] += net.weights1[d1][d2][k];
                }
            }
        }
    }
   
    // 中間層
    int hidden[H_PARAMS2] = {0};
    for (int i = 0; i < H_PARAMS2; i++) {
        for (int j = 0; j < H_PARAMS1; j++) {
            hidden[i] += input[j] * net.weights2[j][i];
        }
        hidden[i] += net.biases2[i];
        hidden[i] = max(0, hidden[i]); // ReLU
    }
   
    // 出力層
    int score = net.bias3;
    for (int i = 0; i < H_PARAMS2; i++) {
        score += hidden[i] * net.weights3[i];
    }
   
    return score;
}

// --- ビームサーチ ---
Action BEAM_SEARCH(F_T f_field[ROW][COL]) {
    int stop = 0;
    int p_maxcombo[DROP+1] = {0};
    int drop[DROP + 1] = { 0 };
    for (int row = 0; row < ROW; row++) {
        for (int col = 0; col < COL; col++) {
            if (1 <= f_field[row][col] && f_field[row][col] <= DROP) {
                drop[f_field[row][col]]++;
            }
        }
    }
    for (int i = 1; i <= DROP; i++) {
        stop += drop[i] / 3;
        p_maxcombo[i]=drop[i]/3;
    }
    MAXCOMBO += (double)stop;

    vector<node> dque;
    double st;
    dque.clear();
   
    // 1手目展開
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            node cand;
            cand.nowR = i;
            cand.nowC = j;
            cand.prev = -1;
            cand.first_te = (T_T)YX(i, j);
            for (int trn = 0; trn <= TRN/21; trn++) cand.movei[trn] = 0ll;
           
            F_T ff_field[ROW][COL];
            memcpy(ff_field,f_field,sizeof(ff_field));
            sc cmb;
            ll ha;
            cand.prev_score=sum_e2(ff_field,&cmb,&ha,p_maxcombo);
            //cand.score=NNUE_init_score(ff_field);
            cand.improving=0;
            cand.hash=ha;
            dque.push_back(cand);
        }
    }

    int dx[DIR] = { -1, 0,0,1 };
    int dy[DIR] = { 0,-1,1,0 };
    Action bestAction;
    int maxValue = 0;
    bestAction.maxcombo = stop;

    unordered_map<ll, bool> checkNodeList[ROW*COL];

    // ビーム探索ループ
    for (int i = 0; i < MAX_TURN; i++) {
        int ks = (int)dque.size();
       
        #pragma omp parallel for private(st)
        for (int k = 0; k < ks; k++) {
            node temp = dque[k];
            F_T temp_field[ROW][COL];
            memcpy(temp_field, f_field, sizeof(temp_field));
            operation(temp_field, temp.first_te,temp.movei);
           
            for (int j = 0; j < DIR; j++) {
                node cand = temp;
                if (0 <= cand.nowC + dx[j] && cand.nowC + dx[j] < COL &&
                    0 <= cand.nowR + dy[j] && cand.nowR + dy[j] < ROW) {
                    if (cand.prev + j != 3) {
                        int ny=cand.nowR + dy[j];
                        int nx=cand.nowC + dx[j];
                        F_T field[ROW][COL];
                        memcpy(field,temp_field,sizeof(temp_field));
                        F_T tmp=field[cand.nowR][cand.nowC];
                        int c1=(int)field[cand.nowR][cand.nowC];
                        int c2=(int)field[ny][nx];
                        cand.hash^=(zoblish_field[cand.nowR][cand.nowC][tmp])^(zoblish_field[ny][nx][field[ny][nx]]);
                        cand.hash^=(zoblish_field[cand.nowR][cand.nowC][field[ny][nx]])^(zoblish_field[ny][nx][tmp]);
                        field[cand.nowR][cand.nowC]=field[ny][nx];
                        field[ny][nx]=tmp;
                        cand.nowC += dx[j];
                        cand.nowR += dy[j];
                        cand.movei[i/21] |= (((ll)(j+1))<<((3*i)%63));
                        //cand.score += NNUE_score(field,c1,c2); // NNUE評価
                        cand.score = NNUE_init_score(field);
                        sc cmb;
                        ll ha;
                        int di=sum_e2(field,&cmb,&ha,p_maxcombo);
                        cand.combo = cmb; // 仮
                        cand.prev = j;
                        fff[(4 * k) + j] = cand;
                    } else {
                        cand.combo = -1;
                        fff[(4 * k) + j] = cand;
                    }
                } else {
                    cand.combo = -1;
                    fff[(4 * k) + j] = cand;
                }
            }
        }

        dque.clear();
        vector<pair<int,int> >vec;
        int ks2 = 0;
        for (int j = 0; j < 4 * ks; j++) {
            if (fff[j].combo != -1) {
                if(fff[j].score>fff[j].prev_score){fff[j].improving=fff[j].improving+1;}
                fff[j].prev_score=fff[j].score;
                int sc=fff[j].score+(BONUS*fff[j].improving)+(fff[j].nowR*3);
                vec.push_back(make_pair(-sc,j));    
                ks2++;
            }
        }
        sort(vec.begin(),vec.end());
        int push_node=0;
        for (int j = 0; push_node < BEAM_WIDTH; j++) {
            if((int)vec.size()<=j){break;}
            int v=vec[j].second;
            node temp = fff[v];
           
            // 理論値コンボチェックは省略(NNUE学習中は評価値のみで走る)
           
            if (i < MAX_TURN - 1) {
                int pos=(temp.nowR*COL)+temp.nowC;
                if(!checkNodeList[pos][temp.hash]){
                    checkNodeList[pos][temp.hash]=true;
                    dque.push_back(temp);
                    push_node++;
                }
            }
            // 暫定ベスト更新
            if(maxValue<(int)temp.combo) {
                maxValue=(int)temp.combo;
                bestAction.first_te = temp.first_te;
                memcpy(bestAction.moving, temp.movei, sizeof(temp.movei));
            }
        }
        if(push_node==0) break;
    }
    return bestAction;
}

// --- 評価バッチ実行 ---
double run_batch(int batch_size) {
    double total_score = 0;
    for (int i = 0; i < batch_size; i++) {
        // 進捗表示
        printf("\r   Batch: %d/%d ", i+1, batch_size);
        fflush(stdout);
       
        F_T f_field[ROW][COL], field[ROW][COL];
        init(f_field); set(f_field, 0);
        Action tmp = BEAM_SEARCH(f_field);
        int path_length=0;
		for (int j = 0; j <= TRN/21; j++) {
			if (tmp.moving[j] == 0ll) { break; }
			for(int k=0;k<21;k++){
			int dir = (int)(7ll&(tmp.moving[j]>>(3*k)));
			if (dir==0){break;}
			path_length++;
			}
		}
        memcpy(field, f_field, sizeof(f_field));
        operation(field, tmp.first_te, tmp.moving);
        int combo = sum_e(field);
       
        if (tmp.maxcombo > 0) {
            double combo_weight = 1.0;
            double length_penalty = 0.001;
            double combo_score = (double)combo / (double)tmp.maxcombo;
            double length_score = (double)path_length * length_penalty;
            total_score += (combo_score - length_score);
        }
    }
    printf("\n");
    return total_score / batch_size;
}

// --- SPSA学習関数 ---
void SA(int type) {
    printf("--- Start SA Iteration %d ---\n", type);
   
    double learning_rate = 1.0;
    double c = 1.0;              
    int batch_size = BATCH;      

    static bool initialized = false;
    if (!initialized) {
        printf("Initializing Net_F from Net...\n");
        // 初期化ループ: D_SIZEまで回す
        for (int i = 0; i < D_SIZE; i++)
            for (int j = 0; j < D_SIZE; j++)
                for (int k = 0; k < H_PARAMS1; k++)
                    net_f.weights1[i][j][k] = (double)net.weights1[i][j][k];

        for (int i = 0; i < H_PARAMS1; i++)
            for (int j = 0; j < H_PARAMS2; j++)
                net_f.weights2[i][j] = (double)net.weights2[i][j];

        for (int i = 0; i < H_PARAMS2; i++) {
            net_f.biases2[i] = (double)net.biases2[i];
            net_f.weights3[i] = (double)net.weights3[i];
        }
        net_f.bias3 = (double)net.bias3;
        initialized = true;
    }

    // 1. Delta生成
    for (int i = 0; i < D_SIZE; i++)
        for (int j = 0; j < D_SIZE; j++)
            for (int k = 0; k < H_PARAMS1; k++)
                delta.weights1[i][j][k] = (rnd(0, 1) == 0) ? 1 : -1;

    for (int i = 0; i < H_PARAMS1; i++)
        for (int j = 0; j < H_PARAMS2; j++)
            delta.weights2[i][j] = (rnd(0, 1) == 0) ? 1 : -1;

    for (int i = 0; i < H_PARAMS2; i++) {
        delta.biases2[i] = (rnd(0, 1) == 0) ? 1 : -1;
        delta.weights3[i] = (rnd(0, 1) == 0) ? 1 : -1;
    }
    delta.bias3 = (rnd(0, 1) == 0) ? 1 : -1;

    // 2. Plus評価
    #define APPLY_DELTA(VAL, D_VAL, SIGN) ((int)round(VAL + (SIGN * c * D_VAL)))
   
    for (int i = 0; i < D_SIZE; i++)
        for (int j = 0; j < D_SIZE; j++)
            for (int k = 0; k < H_PARAMS1; k++)
                net.weights1[i][j][k] = APPLY_DELTA(net_f.weights1[i][j][k], delta.weights1[i][j][k], 1);
   
    for (int i = 0; i < H_PARAMS1; i++)
        for (int j = 0; j < H_PARAMS2; j++)
            net.weights2[i][j] = APPLY_DELTA(net_f.weights2[i][j], delta.weights2[i][j], 1);
    for (int i = 0; i < H_PARAMS2; i++) {
        net.biases2[i] = APPLY_DELTA(net_f.biases2[i], delta.biases2[i], 1);
        net.weights3[i] = APPLY_DELTA(net_f.weights3[i], delta.weights3[i], 1);
    }
    net.bias3 = APPLY_DELTA(net_f.bias3, delta.bias3, 1);

    printf("Evaluating Plus (+)...\n");
    double score_plus = run_batch(batch_size);

    // 3. Minus評価
    for (int i = 0; i < D_SIZE; i++)
        for (int j = 0; j < D_SIZE; j++)
            for (int k = 0; k < H_PARAMS1; k++)
                net.weights1[i][j][k] = APPLY_DELTA(net_f.weights1[i][j][k], delta.weights1[i][j][k], -1);
   
    for (int i = 0; i < H_PARAMS1; i++)
        for (int j = 0; j < H_PARAMS2; j++)
            net.weights2[i][j] = APPLY_DELTA(net_f.weights2[i][j], delta.weights2[i][j], -1);
    for (int i = 0; i < H_PARAMS2; i++) {
        net.biases2[i] = APPLY_DELTA(net_f.biases2[i], delta.biases2[i], -1);
        net.weights3[i] = APPLY_DELTA(net_f.weights3[i], delta.weights3[i], -1);
    }
    net.bias3 = APPLY_DELTA(net_f.bias3, delta.bias3, -1);

    printf("Evaluating Minus (-)...\n");
    double score_minus = run_batch(batch_size);

    // 4. 更新
    double grad_estimate = (score_plus - score_minus) / (2.0 * c);
    double step = learning_rate * grad_estimate;

    for (int i = 0; i < D_SIZE; i++)
        for (int j = 0; j < D_SIZE; j++)
            for (int k = 0; k < H_PARAMS1; k++)
                net_f.weights1[i][j][k] += step * delta.weights1[i][j][k];

    for (int i = 0; i < H_PARAMS1; i++)
        for (int j = 0; j < H_PARAMS2; j++)
            net_f.weights2[i][j] += step * delta.weights2[i][j];

    for (int i = 0; i < H_PARAMS2; i++) {
        net_f.biases2[i] += step * delta.biases2[i];
        net_f.weights3[i] += step * delta.weights3[i];
    }
    net_f.bias3 += step * delta.bias3;

    sync_weights();
    win += (score_plus > max_avg || score_minus > max_avg) ? 1 : 0;
    if (score_plus > max_avg) max_avg = score_plus;
    if (score_minus > max_avg) max_avg = score_minus;

    printf("Iter %d: Score+ = %.4f, Score- = %.4f, Diff = %.4f, Step = %.6f\n",
             type, score_plus, score_minus, score_plus - score_minus, step);
}

// --- 保存関数 ---
bool saveNNUE(const NNUE& net, const NNUE_F& net_f, const NNUE_Delta& delta, const string& filename) {
    ofstream ofs(filename);
    if (!ofs) return false;

    ofs << "===NET===\n";
    for (int i = 0; i < D_SIZE; ++i) {
        for (int j = 0; j < D_SIZE; ++j) {
            for (int k = 0; k < H_PARAMS1; ++k) ofs << net.weights1[i][j][k] << ' ';
            ofs << '\n';
        }
    }
    for (int i = 0; i < H_PARAMS1; ++i) {
        for (int j = 0; j < H_PARAMS2; ++j) ofs << net.weights2[i][j] << ' ';
        ofs << '\n';
    }
    for (int i = 0; i < H_PARAMS2; ++i) ofs << net.biases2[i] << ' ';
    ofs << '\n';
    for (int i = 0; i < H_PARAMS2; ++i) ofs << net.weights3[i] << ' ';
    ofs << '\n';
    ofs << net.bias3 << '\n';

    ofs << "===NET_F===\n";
    ofs.precision(10);
    for (int i = 0; i < D_SIZE; ++i) {
        for (int j = 0; j < D_SIZE; ++j) {
            for (int k = 0; k < H_PARAMS1; ++k) ofs << net_f.weights1[i][j][k] << ' ';
            ofs << '\n';
        }
    }
    for (int i = 0; i < H_PARAMS1; ++i) {
        for (int j = 0; j < H_PARAMS2; ++j) ofs << net_f.weights2[i][j] << ' ';
        ofs << '\n';
    }
    for (int i = 0; i < H_PARAMS2; ++i) ofs << net_f.biases2[i] << ' ';
    ofs << '\n';
    for (int i = 0; i < H_PARAMS2; ++i) ofs << net_f.weights3[i] << ' ';
    ofs << '\n';
    ofs << net_f.bias3 << '\n';

    ofs << "===DELTA===\n";
    for (int i = 0; i < D_SIZE; ++i) {
        for (int j = 0; j < D_SIZE; ++j) {
            for (int k = 0; k < H_PARAMS1; ++k) ofs << (int)delta.weights1[i][j][k] << ' ';
            ofs << '\n';
        }
    }
    for (int i = 0; i < H_PARAMS1; ++i) {
        for (int j = 0; j < H_PARAMS2; ++j) ofs << (int)delta.weights2[i][j] << ' ';
        ofs << '\n';
    }
    for (int i = 0; i < H_PARAMS2; ++i) ofs << (int)delta.biases2[i] << ' ';
    ofs << '\n';
    for (int i = 0; i < H_PARAMS2; ++i) ofs << (int)delta.weights3[i] << ' ';
    ofs << '\n';
    ofs << (int)delta.bias3 << '\n';

    return true;
}

// --- 読み込み関数 ---
bool loadNNUE(NNUE& net, NNUE_F& net_f, NNUE_Delta& delta, const std::string& filename) {
    std::ifstream ifs(filename.c_str());
    if(!ifs) return false;

    std::string marker;
    int temp_int;

    if (!(ifs >> marker) || marker != "===NET===") return false;
    for (int i = 0; i < D_SIZE; ++i) {
        for (int j = 0; j < D_SIZE; ++j) {
            for (int k = 0; k < H_PARAMS1; ++k) if (!(ifs >> net.weights1[i][j][k])) return false;
        }
    }
    for (int i = 0; i < H_PARAMS1; ++i) {
        for (int j = 0; j < H_PARAMS2; ++j) if (!(ifs >> net.weights2[i][j])) return false;
    }
    for (int i = 0; i < H_PARAMS2; ++i) if (!(ifs >> net.biases2[i])) return false;
    for (int i = 0; i < H_PARAMS2; ++i) if (!(ifs >> net.weights3[i])) return false;
    if (!(ifs >> net.bias3)) return false;

    if (!(ifs >> marker) || marker != "===NET_F===") return false;
    for (int i = 0; i < D_SIZE; ++i) {
        for (int j = 0; j < D_SIZE; ++j) {
            for (int k = 0; k < H_PARAMS1; ++k) if (!(ifs >> net_f.weights1[i][j][k])) return false;
        }
    }
    for (int i = 0; i < H_PARAMS1; ++i) {
        for (int j = 0; j < H_PARAMS2; ++j) if (!(ifs >> net_f.weights2[i][j])) return false;
    }
    for (int i = 0; i < H_PARAMS2; ++i) if (!(ifs >> net_f.biases2[i])) return false;
    for (int i = 0; i < H_PARAMS2; ++i) if (!(ifs >> net_f.weights3[i])) return false;
    if (!(ifs >> net_f.bias3)) return false;

    if (!(ifs >> marker) || marker != "===DELTA===") return false;
    for (int i = 0; i < D_SIZE; ++i) {
        for (int j = 0; j < D_SIZE; ++j) {
            for (int k = 0; k < H_PARAMS1; ++k) {
                if (!(ifs >> temp_int)) return false;
                delta.weights1[i][j][k] = (sc)temp_int;
            }
        }
    }
    for (int i = 0; i < H_PARAMS1; ++i) {
        for (int j = 0; j < H_PARAMS2; ++j) {
            if (!(ifs >> temp_int)) return false;
            delta.weights2[i][j] = (sc)temp_int;
        }
    }
    for (int i = 0; i < H_PARAMS2; ++i) {
        if (!(ifs >> temp_int)) return false;
        delta.biases2[i] = (sc)temp_int;
    }
    for (int i = 0; i < H_PARAMS2; ++i) {
        if (!(ifs >> temp_int)) return false;
        delta.weights3[i] = (sc)temp_int;
    }
    if (!(ifs >> temp_int)) return false;
    delta.bias3 = (sc)temp_int;

    return true;
}

// --- その他の関数実装 ---
void show_field(F_T field[ROW][COL]) {
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) printf("%d", field[i][j]);
        printf("\n");
    }
}
void fall(int x,int h,F_T field[ROW][COL]) {
    int tgt;
    for (tgt = ROW - 1; tgt >= h && field[tgt][x] != 0; tgt--);
    for (int i = tgt - 1; i >= h; i--) {
        if (field[i][x] != 0) {
            F_T c = field[i][x];
            field[i][x] = 0;
            field[tgt][x] = c;
            tgt--;
        }
    }
}
void init(F_T field[ROW][COL]) { set(field, !0); }
void set(F_T field[ROW][COL], int force) {
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            if (field[i][j] == 0 || force) field[i][j] = (F_T)rnd(force ? 0 : 1, DROP);
        }
    }
}
int chain(int nrw, int ncl, F_T d, F_T field[ROW][COL], F_T chkflag[ROW][COL], F_T delflag[ROW][COL]) {
    int count = 0;
    #define CHK_CF(Y,X) (field[Y][X] == d && chkflag[Y][X]==0 && delflag[Y][X] > 0)
    if (CHK_CF(nrw, ncl)) {
        ++count; chkflag[nrw][ncl]=1;
        if (0 < nrw && CHK_CF(nrw - 1, ncl)) count += chain(nrw - 1, ncl, d, field, chkflag, delflag);
        if (nrw < ROW - 1 && CHK_CF(nrw + 1, ncl)) count += chain(nrw + 1, ncl, d, field, chkflag, delflag);
        if (0 < ncl && CHK_CF(nrw, ncl - 1)) count += chain(nrw, ncl - 1, d, field, chkflag, delflag);
        if (ncl < COL - 1 && CHK_CF(nrw, ncl + 1)) count += chain(nrw, ncl + 1, d, field, chkflag, delflag);
    }
    return count;
}
int evaluate(F_T field[ROW][COL], int flag) {
    int combo = 0;
    while (1) {
        int cmb = 0;
        F_T chkflag[ROW][COL]={0}, delflag[ROW][COL]={0}, GetHeight[COL];
        for (int row = 0; row < ROW; row++) {
            for (int col = 0; col < COL; col++) {
                F_T num=field[row][col];
                if(row==0) GetHeight[col]=(F_T)ROW;
                if(num>0 && GetHeight[col]==(F_T)ROW) GetHeight[col]=(F_T)row;
                if (col <= COL - 3 && num == field[row][col + 1] && num == field[row][col + 2] && num > 0) {
                    delflag[row][col]=1; delflag[row][col+1]=1; delflag[row][col+2]=1;
                }
                if (row <= ROW - 3 && num == field[row + 1][col] && num == field[row + 2][col] && num > 0) {
                    delflag[row][col]=1; delflag[row+1][col]=1; delflag[row+2][col]=1;
                }
            }
        }
        for (int row = 0; row < ROW; row++) {
            for (int col = 0; col < COL; col++) {
                if (delflag[row][col] > 0) {
                    if (chain(row, col, field[row][col], field, chkflag, delflag) >= 3) cmb++;
                }
            }
        }
        combo += cmb;
        if (cmb == 0 || 0 == (flag & EVAL_COMBO)) break;
        for (int row = 0; row < ROW; row++) for (int col = 0; col < COL; col++) if (delflag[row][col]> 0) field[row][col] = 0;
        if (flag & EVAL_FALL) for(int x=0;x<COL;x++) fall(x,GetHeight[x],field);
        if (flag & EVAL_SET) set(field, 0);
    }
    return combo;
}
int evaluate2(F_T field[ROW][COL], int flag, sc* combo, ll* hash,int p_maxcombo[DROP+1]) {
    int ev = 0; *combo = 0; ll ha=0; int oti = 0; int d_maxcombo[DROP+1]={0};
    while (1) {
        int cmb = 0, cmb2 = 0;
        F_T chkflag[ROW][COL]={0}, delflag[ROW][COL]={0}, GetHeight[COL];
        int cnt_drop[DROP+1]={0}, right[DROP+1], left[DROP+1];
        for(int i=0;i<=DROP;i++){ right[i]=-1; left[i]=COL; }
        for (int row = 0; row < ROW; row++) {
            for (int col = 0; col < COL; col++) {
                F_T num = field[row][col];
                cnt_drop[(int)num]++;
                if(row==0) GetHeight[col]=(F_T)ROW;
                if(num>0 && GetHeight[col]==(F_T)ROW) GetHeight[col]=(F_T)row;
                if(oti==0) ha ^= zoblish_field[row][col][(int)num];
                if (col <= COL - 3 && num == field[row][col + 1] && num == field[row][col + 2] && num > 0) {
                    delflag[row][col]=1; delflag[row][col+1]=1; delflag[row][col+2]=1;
                }
                if (row <= ROW - 3 && num == field[row + 1][col] && num == field[row + 2][col] && num > 0) {
                    delflag[row][col]=1; delflag[row+1][col]=1; delflag[row+2][col]=1;
                }
            }
        }
        F_T erase_x[COL]={0};
        for (int row = 0; row < ROW; row++) {
            for (int col = 0; col < COL; col++) {
                if (delflag[row][col]>0) {
                    int c = chain(row, col, field[row][col], field, chkflag, delflag);
                    if (c >= 3) {
                        cmb++;
                        if (c == 3) cmb2 += 30; else cmb2 += 20;
                        d_maxcombo[(int)field[row][col]]++;
                    }
                    field[row][col]=0; erase_x[col]=1;
                } else {
                    right[(int)field[row][col]]=max(right[(int)field[row][col]],col);
                    left[(int)field[row][col]]=min(left[(int)field[row][col]],col);
                }
            }
        }
        for(int i=1;i<=DROP;i++){
            if(right[i]!=-1&&left[i]!=COL&&cnt_drop[i]>=3&&p_maxcombo[i]!=d_maxcombo[i]) cmb2-=right[i]-left[i];
        }
        *combo += cmb; ev += cmb2;
        if (cmb == 0 || 0 == (flag & EVAL_COMBO)) break;
        oti++;
        if (flag & EVAL_FALL) for(int x=0;x<COL;x++) if(erase_x[x]==1) fall(x,GetHeight[x],field);
        if (flag & EVAL_SET) set(field, 0);
    }
    ev += oti; *hash=ha; return ev;
}
int sum_e2(F_T field[ROW][COL], sc* combo, ll* hash,int p_maxcombo[DROP+1]) { return evaluate2(field, EVAL_FALL | EVAL_COMBO, combo,hash,p_maxcombo); }
int sum_e(F_T field[ROW][COL]) { return evaluate(field, EVAL_FALL | EVAL_COMBO); }
int sum_evaluate(F_T field[ROW][COL]) { return evaluate(field, EVAL_FS | EVAL_COMBO); }
void operation(F_T field[ROW][COL], T_T first_te,ll route[(TRN/21)+1]) {
    int prw = (int)YY(first_te), pcl = (int)XX(first_te), i,j;
    int dx[DIR] = { -1, 0,0,1 }, dy[DIR] = { 0,-1,1,0 };
    for (i = 0; i <= TRN/21; i++) {
        if (route[i] == 0ll) break;
        for(j=0;j<21;j++){
            int dir = (int)(7ll&(route[i]>>(3*j)));
            if(dir==0) break;
            int row=prw+dy[dir-1], col=pcl+dx[dir-1];
            F_T c = field[prw][pcl]; field[prw][pcl] = field[row][col]; field[row][col] = c;
            prw = row; pcl = col;
        }
    }
}
int rnd(int mini, int maxi) {
    static mt19937 mt((int)time(0));
    uniform_int_distribution<int> dice(mini, maxi);
    return dice(mt);
}
ll xor128() {
    static unsigned long long rx = 123456789, ry = 362436069, rz = 521288629, rw = 88675123;
    ll rt = (rx ^ (rx << 11)); rx = ry; ry = rz; rz = rw;
    return (rw = (rw ^ (rw >> 19)) ^ (rt ^ (rt >> 8)));
}

int main() {
    for(int i=0;i<ROW;++i) for(int j=0;j<COL;++j) for(int k=0;k<=DROP;k++) zoblish_field[i][j][k]=xor128();

    bool ln=loadNNUE(net, net_f, delta, "all_nnue.txt");
    if(!ln) printf("New Training Session (Weights initialized to random/zero)\n");
    else printf("Loaded existing weights.\n");

    bool start_test=false;
    if(start_test){
        ifstream myf ("data.txt");
        if(myf.is_open()) {
            printf("Loading teacher data...\n");
            string ls;
            while(getline(myf,ls)){
                string parent="", child="";
                bool slash=false;
                for(int i=0;i<(int)ls.size();i++){
                    if(ls[i]=='/'){slash=true;continue;}
                    if(slash) child+=ls[i]; else parent+=ls[i];
                }
                int counter=0; string xz[3]={"","",""};
                for(int i=0;i<(int)parent.size();i++){
                    if(parent[i]==','){counter++;continue;}
                    xz[counter]+=parent[i];
                }
                // data.txtの読み込み時もオフセットを考慮
                for(int k=0;k<H_PARAMS1;k++){
                    int d1_idx = stoi(xz[1]) + D_OFFSET;
                    int d2_idx = stoi(xz[2]) + D_OFFSET;
                    if (d1_idx >= 0 && d1_idx < D_SIZE && d2_idx >= 0 && d2_idx < D_SIZE) {
                        net.weights1[d1_idx][d2_idx][k]=stoi(child);
                        net_f.weights1[d1_idx][d2_idx][k]=(double)stoi(child); // net_fにも反映
                    }
                }
            }
            myf.close();
            sync_weights(); // int版へ同期
        }
    }

    if(change_weight){
        for(int i=0;i<TEST;i++){
            SA(i);
            if(i%5==0){
                if(saveNNUE(net, net_f, delta, "all_nnue.txt")) printf("--- Saved ---\n");
            }
        }
    }
   
    // --- 最終テスト ---
    return 0; // 学習だけならここで終了でOK
}
