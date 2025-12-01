/*
  g++ -O2 -std=c++11 -fopenmp -mbmi2 xxx.cpp -o xxx
  
  [必須ファイル]
  - train.txt: 盤面(30文字) + "," + スコア
  - data.txt : 色,距離1,距離2 + "/" + スコア (3駒関係の重み)
*/

#pragma warning(disable:4710)
#pragma warning(disable:4711)
#pragma warning(disable:4820)
#pragma GCC target ("sse4.2,bmi2")
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <immintrin.h>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// --- 定数設定 ---
#define ROW 5           // 縦 (MLx.cppと共通)
#define COL 6           // 横 (MLx.cppと共通)
#define DROP 6          // NN学習は標準的な6色ドロップで行う
#define TRN 150         // 手数 (MLx.cppと共通)
#define MAX_TURN 150    // 最大ルート長 (MLx.cppと共通)
#define BEAM_WIDTH 10000
#define NODE_SIZE 5000
#define TEST 10

// NNハイパーパラメータ
#define H_PARAMS 512    // 隠れ層のニューロン数
#define OUTPUT 1
#define LEARNING_RATE 0.00005 // 学習率
#define MOMENTUM 0.9    // モーメンタム係数
#define EPOCHS 3        

// 3駒関係の特徴量サイズ
#define MAX_PATTERNS 2406 // 確保する最大パターン数
#define AUX_FEATURES DROP
#define TOTAL_NN_INPUT_SIZE (MAX_PATTERNS + AUX_FEATURES)

// 型定義 (MLx.cppと共通)
typedef char F_T;
typedef char T_T;
typedef signed char sc;
typedef unsigned char uc;
typedef unsigned long long ll;

// 方向・座標計算 (MLx.cppと共通)
#define DIR 4
#define XX(PT) ((PT)&15)
#define YY(PT) XX((PT)>>4)
#define YX(Y,X) ((Y)<<4|(X))

enum { EVAL_NONE = 0, EVAL_FALL, EVAL_SET, EVAL_FS, EVAL_COMBO };

// グローバル変数
ll zoblish_field[ROW][COL][DROP+1];
ll sqBB[64]; ll file_bb[COL]; int table[64]; ll fill_64[64];
int ac=0;
int wa=0;

// --- 3駒関係パターン管理 ---
struct Pattern {
    int color;
    int d1; // 距離1
    int d2; // 距離2
    int weight; 
};
vector<Pattern> patterns;
int pattern_map[DROP + 1][ROW*COL][ROW*COL]; 
int actual_input_size = 0; 

// --- NNの重み (OpenMP安全のためグローバルに配置) ---
double w1[TOTAL_NN_INPUT_SIZE][H_PARAMS];
double w1_v[TOTAL_NN_INPUT_SIZE][H_PARAMS];
double b1[H_PARAMS], b1_v[H_PARAMS];

double w2[H_PARAMS][H_PARAMS];
double w2_v[H_PARAMS][H_PARAMS];
double b2[H_PARAMS], b2_v[H_PARAMS];

double w3[OUTPUT][H_PARAMS];
double w3_v[OUTPUT][H_PARAMS];
double b3[OUTPUT], b3_v[OUTPUT];

bool is_test = false;
bool learn = false;

// --- 探索用構造体 (MLx.cppと共通) ---
struct node { 
    T_T first_te; 
    ll movei[(TRN / 21) + 1]; // ルート情報
    int score; 
    sc combo; sc nowC; sc nowR; sc prev; ll hash; 
    node() { score = 0; prev = -1; memset(movei,0,sizeof(movei)); } 
    bool operator < (const node& n)const { return score < n.score; } 
};
struct Action { 
    T_T first_te; 
    int score; int maxcombo; 
    ll moving[(TRN / 21) + 1]; 
    Action() { score = 0; memset(moving,0,sizeof(moving)); } 
};
// 探索結果を格納するための配列 (MLx.cppと共通)
node fff[NODE_SIZE];

// --- ヘルパー関数 ---
ll xor128() {
    static unsigned long long x=123456789, y=362436069, z=521288629, w=88675123;
    unsigned long long t = (x^(x<<11));
    x=y; y=z; z=w;
    return ( w = (w^(w>>19))^(t^(t>>8)) );
}
double gaussian_rnd(double mean, double stddev) {
    static mt19937 mt((int)time(0));
    normal_distribution<double> dist(mean, stddev);
    return dist(mt);
}
int rnd(int mini, int maxi) {
    static mt19937 mt((int)time(0));
    uniform_int_distribution<int> dice(mini, maxi);
    return dice(mt);
}
inline double relu(double x) { return x > 0 ? x : 0.0; }
inline double relu_prime(double x) { return x > 0 ? 1.0 : 0.0; }

// 補助関数群 
void init(F_T field[ROW][COL]) { 
    for(int r=0; r<ROW; r++) for(int c=0; c<COL; c++) field[r][c]=(F_T)rnd(1, DROP);
}
void set(F_T field[ROW][COL], int force) { 
       	for (int i = 0; i < ROW; i++) {
		for (int j = 0; j < COL; j++) {
			if (field[i][j] == 0 || force) {//空マスだったらうめる
				field[i][j] = (F_T)rnd(force ? 0 : 1, DROP);//1-DROPの整数乱数
			}
		}
	}
}
void show_field(F_T field[ROW][COL]) {
    for(int r=0; r<ROW; r++) {
        for(int c=0; c<COL; c++) printf("%d", field[r][c]);
        printf("\n");
    }
}

// operation: ルート全体を盤面に反映させる関数 (MLx.cppのシグネチャに準拠)
// この関数は、初期盤面 field に first_teからrouteまでの全移動を適用する役割を担います。
void operation(F_T field[ROW][COL], T_T first_te, ll route[(TRN / 21) + 1]) {
   int prw = (int)YY(first_te), pcl = (int)XX(first_te), i,j;
	int dx[DIR] = { -1, 0,0,1 };
	int dy[DIR] = { 0,-1,1,0 };
	int po=9+(8*(COL-1))+ROW-1;
	for (i = 0; i <= TRN/21; i++) {
		if (route[i] == 0ll) { break; }
		//移動したら、移動前ドロップと移動後ドロップを交換する
		for(j=0;j<21;j++){
		int dir = (int)(7ll&(route[i]>>(3*j)));
		if(dir==0){break;}
		int row=prw+dy[dir-1];
		int col=pcl+dx[dir-1];
		int pre_drop=(int)field[prw][pcl];
		int pre_pos=po-((8*pcl)+prw);
		int next_drop=(int)field[row][col];
		int next_pos=po-((8*col)+row);
		F_T c = field[prw][pcl];
		field[prw][pcl] = field[row][col];
		field[row][col] = c;
		prw = row, pcl = col;
		}//j
	}//i
}

// --- NN関連関数 (省略なし) ---

void load_patterns(const string& filename) {
    ifstream fi(filename);
    if (!fi) {
        cerr << "Error: " << filename << " not found!" << endl;
        exit(1);
    }
    // ... (NN関連の初期化とデータロード)
    for(int i=0; i<=DROP; i++)
        for(int j=0; j<ROW*COL; j++)
            for(int k=0; k<ROW*COL; k++) pattern_map[i][j][k] = -1;

    string line;
    cout << "Loading 3-koma patterns from " << filename << "..." << endl;

    while (getline(fi, line)) {
        int color, d1, d2, score;
        size_t slash = line.find('/');
        if (slash == string::npos) continue;
        
        string left = line.substr(0, slash);
        string right = line.substr(slash + 1);
        
        size_t c1 = left.find(',');
        size_t c2 = left.find(',', c1 + 1);
        
        if (c1 == string::npos || c2 == string::npos) continue;
        
        try {
            color = stoi(left.substr(0, c1));
            d1 = stoi(left.substr(c1 + 1, c2 - (c1 + 1)));
            d2 = stoi(left.substr(c2 + 1));
            score = stoi(right);
        } catch (...) { continue; }

        if (color < 1 || color > DROP) continue;
        if (d1 < 0 || d1 >= ROW*COL || d2 < 0 || d2 >= ROW*COL) continue;

        if (pattern_map[color][d1][d2] == -1) {
            Pattern p = {color, d1, d2, score};
            patterns.push_back(p);
            pattern_map[color][d1][d2] = actual_input_size;
            actual_input_size++;
        }
    }
    fi.close();
    
    if (actual_input_size > MAX_PATTERNS) {
        cerr << "Error: Too many patterns! Increase MAX_PATTERNS." << endl;
        exit(1);
    }
    
    cout << "Loaded " << actual_input_size << " valid patterns (Colors 1-" << DROP << ")." << endl;
}

void count_connected_sets(F_T field[ROW][COL], double counts_out[DROP + 1]) {
    // drop[0]は未使用、drop[1]～drop[DROP]に各色の連結数セットを格納
    
    // 連結判定用の visited 配列
    bool visited[ROW][COL] = {false};
    
    // 各色のセット数をリセット
    for(int c = 1; c <= DROP; ++c) {
        counts_out[c] = 0.0;
    }

    // 盤面全体を走査
    for (int r = 0; r < ROW; ++r) {
        for (int c = 0; c < COL; ++c) {
            if (!visited[r][c]) {
                int color = field[r][c];
                if (color < 1 || color > DROP) continue; // 不正なドロップや空ドロップを無視

                int current_group_size = 0;
                // DFSまたはBFSの代わりに、再帰を使わないスタックで連結成分を探索
                vector<pair<int, int>> stack;
                stack.push_back({r, c});
                visited[r][c] = true;

                while (!stack.empty()) {
                    pair<int, int> current = stack.back();
                    stack.pop_back();
                    current_group_size++;

                    int curr_r = current.first;
                    int curr_c = current.second;

                    // 4方向をチェック
                    int dr[] = {0, 0, 1, -1};
                    int dc[] = {1, -1, 0, 0};

                    for (int i = 0; i < 4; ++i) {
                        int nr = curr_r + dr[i];
                        int nc = curr_c + dc[i];

                        if (nr >= 0 && nr < ROW && nc >= 0 && nc < COL && 
                            !visited[nr][nc] && field[nr][nc] == color) {
                            
                            visited[nr][nc] = true;
                            stack.push_back({nr, nc});
                        }
                    }
                }

                // 3個以上繋がっていたら、セットとしてカウント
                if (current_group_size >= 3) {
                    counts_out[color] += 1.0;
                }
            }
        }
    }
}

void extract_features(F_T field[ROW][COL], double features_out[TOTAL_NN_INPUT_SIZE]) {
    for(int i=0; i<TOTAL_NN_INPUT_SIZE; i++) features_out[i] = 0.0;

    vector<int> pos_by_color[DROP + 1];
    for(int i=0; i<ROW*COL; i++) {
        int color = field[i/COL][i%COL];
        if(color >= 1 && color <= DROP) {
            pos_by_color[color].push_back(i);
        }
    }

    for(int c=1; c<=DROP; c++) {
        const auto& positions = pos_by_color[c];
        int n = (int)positions.size();
        if (n < 3) continue;

        for(int i=0; i<n; i++) {
            for(int j=i+1; j<n; j++) {
                for(int k=j+1; k<n; k++) {
                    int p1 = positions[i];
                    int p2 = positions[j];
                    int p3 = positions[k];
                    
                    int d1 = p2 - p1;
                    int d2 = p3 - p1;
                    
                    if (d1 < ROW*COL && d2 < ROW*COL) {
                        int id = pattern_map[c][d1][d2];
                        if (id != -1 && id < actual_input_size) {
                            features_out[id] += 1.0;
                        }
                    }
                }
            }
        }
    }

    for(int i=0; i<actual_input_size; i++) {
        features_out[i] /= 30.0;
    }
    double connected_counts[DROP + 1]; // 0は未使用
    count_connected_sets(field, connected_counts);
    double stop = 0;
    int drop[DROP + 1] = { 0 };
    for (int row = 0; row < ROW; row++) {
        for (int col = 0; col < COL; col++) {
            if (1 <= field[row][col] && field[row][col] <= DROP) {
                drop[field[row][col]]++;
            }
        }
    }
    for (int i = 1; i <= DROP; i++) {
        stop += (double)(drop[i] / 3);
    }

    for (int c = 1; c <= DROP; ++c) {
        int feature_index = actual_input_size + (c - 1);
        
        // 連結セット数 (例: 3セット) を正規化 (例: 最大コンボ数6で割る)
        // 6コンボは最大値に近いので、6で割って0.0～1.0の範囲に収めます。
        features_out[feature_index] = connected_counts[c] / stop; 
    }
}

double predict_score(F_T field[ROW][COL]) {
    double h1[H_PARAMS], h2[H_PARAMS], out;
    // スレッドセーフなローカル変数
    double features[TOTAL_NN_INPUT_SIZE]; 

    extract_features(field, features);

    // Layer 1
    for (int i = 0; i < H_PARAMS; i++) {
        h1[i] = b1[i];
        for (int k = 0; k < TOTAL_NN_INPUT_SIZE; k++) {
            if (features[k] > 0.0001) { 
                h1[i] += w1[k][i] * features[k];
            }
        }
        h1[i] = relu(h1[i]);
    }

    // Layer 2
    for (int i = 0; i < H_PARAMS; i++) {
        h2[i] = b2[i];
        for (int j = 0; j < H_PARAMS; j++) {
            h2[i] += w2[j][i] * h1[j];
        }
        h2[i] = relu(h2[i]);
    }

    // Output
    out = b3[0];
    for (int j = 0; j < H_PARAMS; j++) {
        out += w3[0][j] * h2[j];
    }
    
    return out;
}

void sgd_train_step(F_T field[ROW][COL], double target) {
    double z1[H_PARAMS], h1[H_PARAMS];
    double z2[H_PARAMS], h2[H_PARAMS];
    double out;
    double features[TOTAL_NN_INPUT_SIZE]; 

    extract_features(field, features);

    // Forward
    for (int i = 0; i < H_PARAMS; i++) {
        z1[i] = b1[i];
        for (int k = 0; k < TOTAL_NN_INPUT_SIZE; k++) {
            if (features[k] > 0.0001) z1[i] += w1[k][i] * features[k];
        }
        h1[i] = relu(z1[i]);
    }
    for (int i = 0; i < H_PARAMS; i++) {
        z2[i] = b2[i];
        for (int j = 0; j < H_PARAMS; j++) z2[i] += w2[j][i] * h1[j];
        h2[i] = relu(z2[i]);
    }
    out = b3[0];
    for (int j = 0; j < H_PARAMS; j++) out += w3[0][j] * h2[j];

    // Backward
    double delta_out = out - target; 

    // Update Layer 3
    for (int j = 0; j < H_PARAMS; j++) {
        double grad = delta_out * h2[j];
        w3_v[0][j] = MOMENTUM * w3_v[0][j] - LEARNING_RATE * grad;
        w3[0][j] += w3_v[0][j];
    }
    b3_v[0] = MOMENTUM * b3_v[0] - LEARNING_RATE * delta_out;
    b3[0] += b3_v[0];

    // Update Layer 2
    double delta_h2[H_PARAMS];
    for (int j = 0; j < H_PARAMS; j++) delta_h2[j] = delta_out * w3[0][j] * relu_prime(z2[j]);
    
    for (int j = 0; j < H_PARAMS; j++) {
        for (int k = 0; k < H_PARAMS; k++) {
            double grad = delta_h2[j] * h1[k];
            w2_v[k][j] = MOMENTUM * w2_v[k][j] - LEARNING_RATE * grad;
            w2[k][j] += w2_v[k][j];
        }
        b2_v[j] = MOMENTUM * b2_v[j] - LEARNING_RATE * delta_h2[j];
        b2[j] += b2_v[j];
    }

    // Update Layer 1
    double delta_h1[H_PARAMS];
    for (int j = 0; j < H_PARAMS; j++) {
        double err = 0;
        for (int k = 0; k < H_PARAMS; k++) err += delta_h2[k] * w2[j][k];
        delta_h1[j] = err * relu_prime(z1[j]);
    }
    
    for (int k = 0; k < TOTAL_NN_INPUT_SIZE; k++) {
        if (features[k] > 0.0001) { 
            for (int j = 0; j < H_PARAMS; j++) {
                double grad = delta_h1[j] * features[k];
                w1_v[k][j] = MOMENTUM * w1_v[k][j] - LEARNING_RATE * grad;
                w1[k][j] += w1_v[k][j];
            }
        }
    }
    for (int j = 0; j < H_PARAMS; j++) {
        b1_v[j] = MOMENTUM * b1_v[j] - LEARNING_RATE * delta_h1[j];
        b1[j] += b1_v[j];
    }
}

void initialize_weights() {
    double scale1 = sqrt(2.0 / actual_input_size);
    for(int k=0; k<TOTAL_NN_INPUT_SIZE; k++) for(int h=0; h<H_PARAMS; h++) w1[k][h] = gaussian_rnd(0, scale1);
    
    double scale2 = sqrt(2.0 / H_PARAMS);
    for(int i=0; i<H_PARAMS; i++) for(int j=0; j<H_PARAMS; j++) w2[i][j] = gaussian_rnd(0, scale2);
    
    double scale3 = sqrt(2.0 / H_PARAMS);
    for(int j=0; j<OUTPUT; j++) for(int i=0; i<H_PARAMS; i++) w3[j][i] = gaussian_rnd(0, scale3);

    memset(b1, 0, sizeof(b1)); memset(b2, 0, sizeof(b2)); memset(b3, 0, sizeof(b3));
    memset(w1_v, 0, sizeof(w1_v)); memset(w2_v, 0, sizeof(w2_v)); memset(w3_v, 0, sizeof(w3_v));

    cout << "Applying Transfer Learning to W1..." << endl;
    double max_pattern_score = 1.0;
    for (int i = 0; i < actual_input_size; i++) { // actual_input_size まで
        if (abs(patterns[i].weight) > max_pattern_score) max_pattern_score = abs(patterns[i].weight);
    }
    
    for (int k = 0; k < actual_input_size; k++) {
        double base_weight = (double)patterns[k].weight / max_pattern_score;
        
        for (int h = 0; h < H_PARAMS; h++) {
            w1[k][h] += base_weight * 0.5;
        }
    }
    cout << "Transfer Learning Complete." << endl;
}

void train_process() {
    vector<pair<string, double>> dataset;
    ifstream fi("train.txt");
    if (!fi) { cout << "train.txt not found." << endl; return; }

    cout << "Loading train data..." << endl;
    string line;
int counter=0;
    while (getline(fi, line)) {
        size_t comma = line.find(',');
        if (comma == string::npos) continue;
        string b_str = line.substr(0, comma);
        int raw_score = 0;
        try {
            raw_score = stoi(line.substr(comma + 1));
        } catch (...) { continue; }
        
        double target = (double)raw_score / 50000.0;
        if(target < 0.0) target = 0.0;
        if(target > 1.0) target = 1.0;
        counter++;
//if(counter>=100000){break;}
        dataset.push_back(make_pair(b_str, target));
    }
    fi.close();
    cout << "Data Size: " << dataset.size() << endl;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        shuffle(dataset.begin(), dataset.end(), mt19937((unsigned int)time(0)));
        
        for (size_t i = 0; i < dataset.size(); i++) {
            const auto& p = dataset[i];
            F_T field[ROW][COL];
            for (int j = 0; j < ROW * COL; j++) field[j/COL][j%COL] = p.first[j] - '0';
            
            double pred = predict_score(field);
            double diff = pred - p.second;
            total_loss += diff * diff;

            sgd_train_step(field, p.second);
        }
        printf("Epoch %d/%d, Loss: %.6f\n", epoch + 1, EPOCHS, total_loss / dataset.size());
    }
}
int chain(int nrw, int ncl, F_T d, F_T field[ROW][COL],
	F_T chkflag[ROW][COL], F_T delflag[ROW][COL]) {
	int count = 0;
#define CHK_CF(Y,X) (field[Y][X] == d && chkflag[Y][X]==0 && delflag[Y][X] > 0)
	//連結している同じ色の消去ドロップが未探索だったら
	if (CHK_CF(nrw, ncl)) {
		++count; //連結ドロップ数の更新
		chkflag[nrw][ncl]=1;//探索済みにする
			//以下上下左右に連結しているドロップを再帰的に探索していく
		if (0 < nrw && CHK_CF(nrw - 1, ncl)) {
			count += chain(nrw - 1, ncl, d, field, chkflag, delflag);
		}
		if (nrw < ROW - 1 && CHK_CF(nrw + 1, ncl)) {
			count += chain(nrw + 1, ncl, d, field, chkflag, delflag);
		}
		if (0 < ncl && CHK_CF(nrw, ncl - 1)) {
			count += chain(nrw, ncl - 1, d, field, chkflag, delflag);
		}
		if (ncl < COL - 1 && CHK_CF(nrw, ncl + 1)) {
			count += chain(nrw, ncl + 1, d, field, chkflag, delflag);
		}
	}
	return count;
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
int evaluate(F_T field[ROW][COL], int flag) {
	int combo = 0;

	while (1) {
		int cmb = 0;
		F_T chkflag[ROW][COL]={0};
		F_T delflag[ROW][COL]={0};
		F_T GetHeight[COL];
		for (int row = 0; row < ROW; row++) {
			for (int col = 0; col < COL; col++) {
				F_T num=field[row][col];
				if(row==0){
				GetHeight[col]=(F_T)ROW;
				}
				if(num>0 && GetHeight[col]==(F_T)ROW){
				GetHeight[col]=(F_T)row;
				}
				if (col <= COL - 3 && num == field[row][col + 1] && num == field[row][col + 2] && num > 0) {
					delflag[row][col]=1;
					delflag[row][col+1]=1;
					delflag[row][col+2]=1;
				}
				if (row <= ROW - 3 && num == field[row + 1][col] && num == field[row + 2][col] && num > 0) {
					delflag[row][col]=1;
					delflag[row+1][col]=1;
					delflag[row+2][col]=1;
				}
			}
		}
		for (int row = 0; row < ROW; row++) {
			for (int col = 0; col < COL; col++) {
				if (delflag[row][col] > 0) {
					if (chain(row, col, field[row][col], field, chkflag, delflag) >= 3) {
						cmb++;
					}
				}
			}
		}
		combo += cmb;
		//コンボが発生しなかったら終了
		if (cmb == 0 || 0 == (flag & EVAL_COMBO)) { break; }
		for (int row = 0; row < ROW; row++) {
			for (int col = 0; col < COL; col++) {
				//コンボになったドロップは空になる
				if (delflag[row][col]> 0) { field[row][col] = 0; }
			}
		}

		if (flag & EVAL_FALL){
		for(int x=0;x<COL;x++){
		fall(x,GetHeight[x],field);
		}
		}//落下処理発生
		if (flag & EVAL_SET){set(field, 0);}//落ちコン発生

	}
	return combo;
}
void make_url(const F_T f_field[ROW][COL], const Action& tmp) {
string layout="";

for(int v=0;v<ROW;v++){
for(int u=0;u<COL;u++){
layout+=to_string(f_field[v][u]-1);
}
}
string route="";
//printf("(x,y)=(%d,%d)", XX(tmp.first_te), YY(tmp.first_te));
int path_length=0;
route+=to_string(XX(tmp.first_te))+to_string(YY(tmp.first_te)+5)+",";
for (int j = 0; j <= TRN/21; j++) {//y座標は下にいくほど大きくなる
if (tmp.moving[j] == 0ll) { break; }
for(int k=0;k<21;k++){
int dir = (int)(7ll&(tmp.moving[j]>>(3*k)));
if (dir==0){break;}
if (dir==1) { route+=to_string(3);}//printf("L"); } //"LEFT"); }
if (dir==2) { route+=to_string(6);}//printf("U"); } //"UP"); }
if (dir==3) { route+=to_string(1);}//printf("D"); } //"DOWN"); }
if (dir==4) { route+=to_string(4);}//printf("R"); } //"RIGHT"); }
path_length++;
}
}
string url="http://serizawa.web5.jp/puzzdra_theory_maker/index.html?layout="+layout+"&route="+route+"&ctwMode=false";
cout<<url<<endl;
printf("\n");
int tgt=0;
string top="";
F_T fff_field[ROW][COL];
memcpy(fff_field,f_field,sizeof(fff_field));
while(1){

if(route[tgt]==','){tgt++;break;}
top+=route[tgt];
tgt++;
}
int pos;
if((int)top.size()==2){int x=top[0]-'0';int y=(top[1]-'0')-5;pos=(y*COL)+x;}
else{int x=top[0]-'0';int y=5;pos=(y*COL)+x;}

for(int j=tgt;j<(int)route.size();j++){
if(route[j]=='3'){swap(fff_field[pos/COL][pos%COL],fff_field[pos/COL][(pos%COL)-1]);pos--;}
if(route[j]=='6'){swap(fff_field[pos/COL][pos%COL],fff_field[(pos/COL)-1][pos%COL]);pos-=COL;}
if(route[j]=='1'){swap(fff_field[pos/COL][pos%COL],fff_field[(pos/COL)+1][pos%COL]);pos+=COL;}
if(route[j]=='4'){swap(fff_field[pos/COL][pos%COL],fff_field[pos/COL][(pos%COL)+1]);pos++;}
}

int cb=evaluate(fff_field, EVAL_FALL | EVAL_COMBO);
int stop = 0;
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
}
cout<<"combo="<<cb<<"/"<<stop<<endl;

if(cb==stop){ac++;}
else{wa++;}
printf("ac=%d/%d\n",ac,(ac+wa));
}

// ----------------------------------------------------------------------
// ★ MLx.cpp 風 ビームサーチ NN評価関数 ★
// ----------------------------------------------------------------------

Action BEAM_SEARCH_NN(F_T f_field[ROW][COL]) {
    vector<node> dque;
    // 初期ノードの生成 (全てのドロップを起点とする)
    for (int r = 0; r < ROW; r++) { 
        for (int c = 0; c < COL; c++) { 
            node cand; 
            cand.nowR = r; 
            cand.nowC = c; 
            cand.first_te = (T_T)YX(r, c); 
            dque.push_back(cand); 
        } 
    }

    int dx[] = { -1, 0, 0, 1 }; // L, U, D, R (dir 1, 2, 3, 4)
    int dy[] = { 0, -1, 1, 0 }; 
    Action best;
    int best_nn_score = -9999999;

    // 探索ループ (MAX_TURNまで)
    for (int i = 0; i < MAX_TURN; i++) {
        int ks = (int)dque.size();
        
        // OpenMPによる並列処理
        #pragma omp parallel for
        for (int k = 0; k < ks; k++) {
            node temp = dque[k];
            
            for (int j = 0; j < DIR; j++) {
                node cand = temp; // 親ノードの情報をコピー (MLxスタイル)
                int nx = cand.nowC + dx[j];
                int ny = cand.nowR + dy[j];

                // 移動先が盤面内かチェック
                if (nx >= 0 && nx < COL && ny >= 0 && ny < ROW) {
                    // 逆方向への即時移動を除外
                    if (cand.prev + j != 3) { 
                        
                        // 次のドロップ位置を更新
                        cand.nowC = nx; cand.nowR = ny;
                        
                        // 移動ルートを記録 (3bitエンコード)
                        cand.movei[i / 21] |= (((ll)(j + 1)) << ((3 * i) % 63));
                        cand.prev = j;

                        // ★ 盤面再構築とNN推論 (このブロックがNN評価のコア) ★
                        F_T next_field[ROW][COL];
                        // 1. 初期盤面をコピー（高速なmemcpyを使用）
                        memcpy(next_field, f_field, sizeof(next_field));
                        
                        // 2. operation関数を呼び出し、ルート全体を盤面に反映させる
                        // これにより next_field が NN 評価に必要な状態になる
                        operation(next_field, cand.first_te, cand.movei); 
                        
                        // 3. NN推論
                        double val = predict_score(next_field);
                        // 正規化されたスコアを100万倍して整数に
                        cand.score = (int)(val * 1000000); 
                        
                        // OpenMPの配列に格納 (インデックス衝突なし)
                        fff[(4 * k) + j] = cand;
                    } else { 
                        // 逆方向の移動は無効なスコアでマーク
                        node d; d.score = -9999999; fff[(4 * k) + j] = d; 
                    }
                } else { 
                    // 盤外への移動は無効なスコアでマーク
                    node d; d.score = -9999999; fff[(4 * k) + j] = d; 
                }
            }
        }
        
        // 選抜処理 (MLxスタイル)
        dque.clear();
        vector<pair<int, int>> vec;
        // 有効なスコアを持つノードのみを収集
        for (int j = 0; j < 4 * ks; j++) if (fff[j].score > -9000000) vec.push_back(make_pair(-fff[j].score, j));
        sort(vec.begin(), vec.end()); // スコアの降順でソート (pairの-scoreで実現)
        
        // ベストスコアの更新
        if (vec.size() > 0) {
            node& top = fff[vec[0].second];
            if (top.score > best_nn_score) { 
                best_nn_score = top.score; 
                best.score = top.score; 
                best.first_te = top.first_te; 
                memcpy(best.moving, top.movei, sizeof(top.movei)); 
            }
        }
        
        // ビーム幅でノードを選抜
        int limit = BEAM_WIDTH;
        for (int j = 0; j < limit && j < (int)vec.size(); j++) dque.push_back(fff[vec[j].second]);
        
        if (dque.empty()) break;
    }
    return best;
}

// ----------------------------------------------------------------------
// main, sub関数 (MLx.cppと共通)
// ----------------------------------------------------------------------

void sub() {
    // 探索コード由来の初期化
    int po = 9 + (8 * (COL - 1)) + ROW - 1;
    for (int i = 0; i < ROW; i++) for (int j = 0; j < COL; j++) sqBB[po-(8*j)-i] |= 1ll<<(po-(8*j)-i);
    ll ha = 0x03F566ED27179461ULL;
    for (int i = 0; i < 64; i++) { table[ha >> 58] = i; ha <<= 1; }
    ll res = 2ll; fill_64[1] = res;
    for (int i = 2; i < 64; i++) { fill_64[i] = res + (1ll << i); res = fill_64[i]; }
    for (int i = 0; i < COL; i++) { for (int j = 0; j < ROW; j++) file_bb[i] |= (1ll << (po - j)); po -= 8; }

    // 1. data.txt から3駒関係パターンをロード
    load_patterns("data.txt");

    // 2. 重みの初期化 & 転移学習
    initialize_weights();

    if (learn) {
        // 3. NN学習
        train_process();
        is_test = true;
    } 
    
    // NNを使用した実戦テスト
    printf("\n--- NN Beam Search Test Start ---\n");
    for (int i = 0; i < TEST; i++) {
        F_T f_field[ROW][COL];
        printf("Test No.%d/%d\n", i + 1, TEST);
        init(f_field); set(f_field, 0);
        
        printf("Initial Field:\n");
        show_field(f_field);
        
        // ビームサーチ実行
        Action tmp = BEAM_SEARCH_NN(f_field);
        
        // 結果出力
        printf("NN Score: %d (Max 1000000)\n", tmp.score);
        make_url(f_field, tmp);
        printf("--------------------------------\n");
    }
}

bool saveParams(const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return false;

    // 固定サイズなのでループで全部書き出す（テキスト）
    for (int i = 0; i < MAX_PATTERNS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            fprintf(fp, "%.17g\n", w1[i][j]);
    for (int i = 0; i < MAX_PATTERNS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            fprintf(fp, "%.17g\n", w1_v[i][j]);

    for (int i = 0; i < H_PARAMS; ++i)
        fprintf(fp, "%.17g\n", b1[i]);
    for (int i = 0; i < H_PARAMS; ++i)
        fprintf(fp, "%.17g\n", b1_v[i]);

    for (int i = 0; i < H_PARAMS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            fprintf(fp, "%.17g\n", w2[i][j]);
    for (int i = 0; i < H_PARAMS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            fprintf(fp, "%.17g\n", w2_v[i][j]);

    for (int i = 0; i < H_PARAMS; ++i)
        fprintf(fp, "%.17g\n", b2[i]);
    for (int i = 0; i < H_PARAMS; ++i)
        fprintf(fp, "%.17g\n", b2_v[i]);

    for (int i = 0; i < OUTPUT; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            fprintf(fp, "%.17g\n", w3[i][j]);
    for (int i = 0; i < OUTPUT; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            fprintf(fp, "%.17g\n", w3_v[i][j]);

    for (int i = 0; i < OUTPUT; ++i)
       fprintf(fp, "%.17g\n", b3[i]);
    for (int i = 0; i < OUTPUT; ++i)
       fprintf(fp, "%.17g\n", b3_v[i]);

    fclose(fp);
    return true;
}

bool loadParams(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return false;

    // 保存と同じ順番で読む
    for (int i = 0; i < MAX_PATTERNS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            if (fscanf(fp, "%lf", &w1[i][j]) != 1) { fclose(fp); return false; }
    for (int i = 0; i < MAX_PATTERNS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            if (fscanf(fp, "%lf", &w1_v[i][j]) != 1) { fclose(fp); return false; }

    for (int i = 0; i < H_PARAMS; ++i)
        if (fscanf(fp, "%lf", &b1[i]) != 1) { fclose(fp); return false; }
    for (int i = 0; i < H_PARAMS; ++i)
        if (fscanf(fp, "%lf", &b1_v[i]) != 1) { fclose(fp); return false; }

    for (int i = 0; i < H_PARAMS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            if (fscanf(fp, "%lf", &w2[i][j]) != 1) { fclose(fp); return false; }
    for (int i = 0; i < H_PARAMS; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            if (fscanf(fp, "%lf", &w2_v[i][j]) != 1) { fclose(fp); return false; }

    for (int i = 0; i < H_PARAMS; ++i)
        if (fscanf(fp, "%lf", &b2[i]) != 1) { fclose(fp); return false; }
    for (int i = 0; i < H_PARAMS; ++i)
        if (fscanf(fp, "%lf", &b2_v[i]) != 1) { fclose(fp); return false; }

    for (int i = 0; i < OUTPUT; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            if (fscanf(fp, "%lf", &w3[i][j]) != 1) { fclose(fp); return false; }
    for (int i = 0; i < OUTPUT; ++i)
        for (int j = 0; j < H_PARAMS; ++j)
            if (fscanf(fp, "%lf", &w3_v[i][j]) != 1) { fclose(fp); return false; }

    for (int i = 0; i < OUTPUT; ++i)
        if (fscanf(fp, "%lf", &b3[i]) != 1) { fclose(fp); return false; }
    for (int i = 0; i < OUTPUT; ++i)
        if (fscanf(fp, "%lf", &b3_v[i]) != 1) { fclose(fp); return false; }

    fclose(fp);
    return true;
}

int main() {
    string sss = "";
    printf("train?(y/n)=");
    cin >> sss;
    learn = (sss == "y" || sss == "Y");
    int i, j, k;
    for(i=0;i<ROW;++i) for(j=0;j<COL;++j) for(k=0;k<=DROP;k++) zoblish_field[i][j][k]=xor128();
    
    #ifdef _OPENMP
    // omp_set_num_threads(8); // 必要に応じてコメントを外してコア数を設定
    #endif
    if(!learn){
    if (!loadParams("params.txt")) {
        printf("load failed\n");
    }
    }
    sub();
    if(learn){
    if (!saveParams("params.txt")) {
    printf("save failed\n");
    }
    }
    return 0;
}
