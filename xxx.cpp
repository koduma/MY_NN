/*
 g++ -O2 -std=c++11 -fopenmp -mbmi2 yyy.cpp -o ML
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
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// --- 定数 ---
#define ROW 5
#define COL 6
#define DROP 6
#define TRN 150
#define MAX_TURN 150
#define BEAM_WIDTH 100
#define PROBLEM 1000
#define BONUS 10
#define NODE_SIZE 3000
#define TEST 100

// NNパラメータ
// 入力: 盤面(30) + 補助特徴量(2: 横連結数, 縦連結数) = 32
#define INPUT_SIZE (ROW*COL)
#define AUX_SIZE 2 
#define TOTAL_INPUT (INPUT_SIZE + AUX_SIZE)

#define H_PARAMS 128    // 隠れ層を少し大きく
#define OUTPUT 1
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9    // 慣性項
#define EPOCHS 50       // エポック数
#define BATCH_SIZE 32   // ミニバッチサイズ

typedef char F_T;
typedef char T_T;
typedef signed char sc;
typedef unsigned char uc;
typedef unsigned long long ll;

// 方向
#define DIR 4
#define XX(PT) ((PT)&15)
#define YY(PT) XX((PT)>>4)
#define YX(Y,X) ((Y)<<4|(X))

enum { EVAL_NONE = 0, EVAL_FALL, EVAL_SET, EVAL_FS, EVAL_COMBO };

ll zoblish_field[ROW][COL][DROP+1];

// --- NNの重みとモーメンタム ---
// Embedding的な第1層: [場所][色][ニューロン]
double w1[INPUT_SIZE][DROP + 1][H_PARAMS]; 
double w1_v[INPUT_SIZE][DROP + 1][H_PARAMS]; // 慣性(Velocity)

// 補助特徴量用の第1層: [特徴][ニューロン]
double w1_aux[AUX_SIZE][H_PARAMS];
double w1_aux_v[AUX_SIZE][H_PARAMS];

double b1[H_PARAMS], b1_v[H_PARAMS]; // Bias 1

double w2[H_PARAMS][H_PARAMS];
double w2_v[H_PARAMS][H_PARAMS];
double b2[H_PARAMS], b2_v[H_PARAMS]; // Bias 2

double w3[OUTPUT][H_PARAMS];
double w3_v[OUTPUT][H_PARAMS];
double b3[OUTPUT], b3_v[OUTPUT];     // Bias 3

bool is_test = false;
bool learn = false;

// --- ビットボード用 ---
ll sqBB[64];
ll file_bb[COL];
int table[64];
ll fill_64[64];

// --- 構造体 ---
struct node {
    T_T first_te;
    ll movei[(TRN / 21) + 1];
    int score; // NNのスコア(x10000した整数)
    sc combo;
    sc nowC; sc nowR;
    sc prev;
    ll hash;
    node() { score = 0; prev = -1; memset(movei,0,sizeof(movei)); }
    bool operator < (const node& n)const { return score < n.score; }
};

struct Action {
    T_T first_te;
    int score;
    int maxcombo;
    ll moving[(TRN / 21) + 1];
    Action() { score = 0; memset(moving,0,sizeof(moving)); }
};

node fff[NODE_SIZE];

// --- 関数プロトタイプ ---
void init(F_T field[ROW][COL]);
void set(F_T field[ROW][COL], int force);
void show_field(F_T field[ROW][COL]);
int rnd(int mini, int maxi);
double gaussian_rnd(double mean, double stddev); // ガウス分布
int chain(int nrw, int ncl, F_T d, F_T field[ROW][COL], F_T chkflag[ROW][COL], F_T delflag[ROW][COL]);
void fall(int x, int h, F_T field[ROW][COL]);
void operation(F_T field[ROW][COL], T_T first_te, ll route[(TRN / 21) + 1], ll dropBB[DROP + 1]);
ll xor128();
ll calc_mask(ll bitboard);
ll fallBB(ll p, ll rest, ll mask);
int evaluate3(ll dropBB[DROP + 1], int flag, sc* combo, int p_maxcombo[DROP + 1]);
void sub();


void init(F_T field[ROW][COL]) { set(field, !0); }
void set(F_T field[ROW][COL], int force) {
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            if (field[i][j] == 0 || force) {
                field[i][j] = (F_T)rnd(1, DROP);
            }
        }
    }
}
void show_field(F_T field[ROW][COL]) {
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) printf("%d", field[i][j]);
        printf("\n");
    }
}

// --- 乱数 ---
int rnd(int mini, int maxi) {
    static mt19937 mt((int)time(0));
    uniform_int_distribution<int> dice(mini, maxi);
    return dice(mt);
}

// Heの初期値用 (正規分布)
double gaussian_rnd(double mean, double stddev) {
    static mt19937 mt((int)time(0));
    normal_distribution<double> dist(mean, stddev);
    return dist(mt);
}

// --- NN関連 (ReLU + Momentum SGD) ---

inline double relu(double x) { return x > 0 ? x : 0.0; }
inline double relu_prime(double x) { return x > 0 ? 1.0 : 0.0; }

// 特徴量抽出: 横連結数と縦連結数を数える (これがNNへの最大のヒントになる)
void extract_features(F_T field[ROW][COL], double aux_out[AUX_SIZE]) {
    int h_conn = 0;
    int v_conn = 0;
    // 横連結
    for(int r=0; r<ROW; r++) {
        for(int c=0; c<COL-1; c++) {
            if(field[r][c] != 0 && field[r][c] == field[r][c+1]) h_conn++;
        }
    }
    // 縦連結
    for(int c=0; c<COL; c++) {
        for(int r=0; r<ROW-1; r++) {
            if(field[r][c] != 0 && field[r][c] == field[r+1][c]) v_conn++;
        }
    }
    // 正規化 (0.0 ~ 1.0くらいに収まるように割る)
    aux_out[0] = (double)h_conn / 15.0; 
    aux_out[1] = (double)v_conn / 15.0;
}

// 推論 (Forward)
double predict_score(F_T field[ROW][COL]) {
    double h1[H_PARAMS], h2[H_PARAMS], out;
    double aux[AUX_SIZE];

    extract_features(field, aux);

    // Layer 1
    for (int i = 0; i < H_PARAMS; i++) {
        h1[i] = b1[i];
        // 盤面情報 (Embedding)
        for (int k = 0; k < INPUT_SIZE; k++) {
            int color = field[k/COL][k%COL];
            if(color >= 1 && color <= DROP)
                h1[i] += w1[k][color][i];
        }
        // 補助特徴量
        for (int k = 0; k < AUX_SIZE; k++) {
            h1[i] += w1_aux[k][i] * aux[k];
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
    
    return out; // 0.0 ~ 1.0 の値を期待
}

// 学習 (Momentum SGD)
void sgd_train_step(F_T field[ROW][COL], double target) {
    double z1[H_PARAMS], h1[H_PARAMS];
    double z2[H_PARAMS], h2[H_PARAMS];
    double out;
    double aux[AUX_SIZE];

    extract_features(field, aux);

    // --- Forward ---
    for (int i = 0; i < H_PARAMS; i++) {
        z1[i] = b1[i];
        for (int k = 0; k < INPUT_SIZE; k++) {
            int color = field[k/COL][k%COL];
            if(color >= 1 && color <= DROP) z1[i] += w1[k][color][i];
        }
        for (int k = 0; k < AUX_SIZE; k++) z1[i] += w1_aux[k][i] * aux[k];
        h1[i] = relu(z1[i]);
    }

    for (int i = 0; i < H_PARAMS; i++) {
        z2[i] = b2[i];
        for (int j = 0; j < H_PARAMS; j++) z2[i] += w2[j][i] * h1[j];
        h2[i] = relu(z2[i]);
    }

    out = b3[0];
    for (int j = 0; j < H_PARAMS; j++) out += w3[0][j] * h2[j];

    // --- Backward ---
    double delta_out = out - target; // MSEの勾配

    // Layer 3 update (Momentum)
    for (int j = 0; j < H_PARAMS; j++) {
        double grad = delta_out * h2[j];
        w3_v[0][j] = MOMENTUM * w3_v[0][j] - LEARNING_RATE * grad;
        w3[0][j] += w3_v[0][j];
    }
    b3_v[0] = MOMENTUM * b3_v[0] - LEARNING_RATE * delta_out;
    b3[0] += b3_v[0];

    // Layer 2 Backprop
    double delta_h2[H_PARAMS];
    for (int j = 0; j < H_PARAMS; j++) {
        delta_h2[j] = delta_out * w3[0][j] * relu_prime(z2[j]);
    }
    // Layer 2 Update
    for (int j = 0; j < H_PARAMS; j++) {
        for (int k = 0; k < H_PARAMS; k++) {
            double grad = delta_h2[j] * h1[k];
            w2_v[k][j] = MOMENTUM * w2_v[k][j] - LEARNING_RATE * grad;
            w2[k][j] += w2_v[k][j];
        }
        b2_v[j] = MOMENTUM * b2_v[j] - LEARNING_RATE * delta_h2[j];
        b2[j] += b2_v[j];
    }

    // Layer 1 Backprop
    double delta_h1[H_PARAMS];
    for (int j = 0; j < H_PARAMS; j++) {
        double error = 0;
        for (int k = 0; k < H_PARAMS; k++) error += delta_h2[k] * w2[j][k];
        delta_h1[j] = error * relu_prime(z1[j]);
    }
    // Layer 1 Update
    for (int k = 0; k < INPUT_SIZE; k++) {
        int color = field[k/COL][k%COL];
        if(color >= 1 && color <= DROP) {
            for (int j = 0; j < H_PARAMS; j++) {
                double grad = delta_h1[j];
                w1_v[k][color][j] = MOMENTUM * w1_v[k][color][j] - LEARNING_RATE * grad;
                w1[k][color][j] += w1_v[k][color][j];
            }
        }
    }
    for (int k = 0; k < AUX_SIZE; k++) {
        for (int j = 0; j < H_PARAMS; j++) {
            double grad = delta_h1[j] * aux[k];
            w1_aux_v[k][j] = MOMENTUM * w1_aux_v[k][j] - LEARNING_RATE * grad;
            w1_aux[k][j] += w1_aux_v[k][j];
        }
    }
    for (int j = 0; j < H_PARAMS; j++) {
        b1_v[j] = MOMENTUM * b1_v[j] - LEARNING_RATE * delta_h1[j];
        b1[j] += b1_v[j];
    }
}

// --- データ管理 ---
void memo_data(F_T field[ROW][COL], int raw_score) {
    ofstream fi("train.txt", ios::app);
    for (int i = 0; i < ROW * COL; i++) fi << (int)field[i / COL][i % COL];
    fi << "," << raw_score << "\n";
    fi.close();
}

void train_process() {
    vector<pair<string, double>> dataset;
    ifstream fi("train.txt");
    string line;
    if (!fi) { cout << "train.txt not found." << endl; return; }

    cout << "Loading data..." << endl;
    while (getline(fi, line)) {
        size_t comma = line.find(',');
        if (comma == string::npos) continue;
        string b_str = line.substr(0, comma);
        int raw_score = stoi(line.substr(comma + 1));
        
        // 重要: スコアの正規化 (0.0 ~ 1.0)
        // 10000は理論値到達のフラグ
        double target = 0.0;
        if(raw_score >= 9900) target = 1.0; 
        else {
            // 2000~3000くらいのスコアを 0.0 ~ 0.8 にマッピング
            // Evaluate3の戻り値はおよそ 2000点 + コンボボーナス
            target = (double)(raw_score - 2000) / 1500.0;
            if(target < 0.0) target = 0.0;
            if(target > 0.9) target = 0.9;
        }
        dataset.push_back(make_pair(b_str, target));
    }
    fi.close();
    
    // データを間引く (10万件に制限)
    if(dataset.size() > 100000) {
        shuffle(dataset.begin(), dataset.end(), mt19937(time(0)));
        dataset.resize(100000);
        cout << "Data reduced to 100,000 for speed." << endl;
    }

    cout << "Training Start. Size: " << dataset.size() << endl;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        shuffle(dataset.begin(), dataset.end(), mt19937(time(0)));
        
        for (const auto& p : dataset) {
            F_T field[ROW][COL];
            for (int i = 0; i < INPUT_SIZE; i++) field[i/COL][i%COL] = p.first[i] - '0';
            
            // 誤差計算 (表示用)
            double pred = predict_score(field);
            total_loss += (pred - p.second) * (pred - p.second);

            // 学習
            sgd_train_step(field, p.second);
        }
        printf("Epoch %d/%d, Loss: %.6f\n", epoch + 1, EPOCHS, total_loss / dataset.size());
        
        // Lossが十分に下がったら終了
        if((total_loss / dataset.size()) < 0.005) break;
    }
}

// --- ビームサーチ ---
// NNを使った推論モード
Action BEAM_SEARCH_NN(F_T f_field[ROW][COL]) {
    int po = 9 + (8 * (COL - 1)) + ROW - 1;
    vector<node> dque;
    
    // 1手目全探索
    for (int r = 0; r < ROW; r++) {
        for (int c = 0; c < COL; c++) {
            node cand;
            cand.nowR = r; cand.nowC = c;
            cand.first_te = (T_T)YX(r, c);
            dque.push_back(cand);
        }
    }

    int dx[] = { -1, 0, 0, 1 };
    int dy[] = { 0, -1, 1, 0 };
    Action best;
    double best_nn_score = -9999.0;

    ll rootBB[DROP + 1] = { 0 };
    for (int r = 0; r < ROW; r++)
        for (int c = 0; c < COL; c++)
            rootBB[f_field[r][c]] |= (1ll << (po - ((8 * c) + r)));

    for (int i = 0; i < MAX_TURN; i++) {
        int ks = (int)dque.size();
        
        // 並列化: NN推論は重いので並列化が効く
        #pragma omp parallel for
        for (int k = 0; k < ks; k++) {
            node temp = dque[k];
            F_T temp_field[ROW][COL];
            ll temp_dropBB[DROP + 1];
            memcpy(temp_field, f_field, sizeof(temp_field));
            memcpy(temp_dropBB, rootBB, sizeof(rootBB));
            operation(temp_field, temp.first_te, temp.movei, temp_dropBB);

            for (int j = 0; j < DIR; j++) {
                node cand = temp;
                int nx = cand.nowC + dx[j];
                int ny = cand.nowR + dy[j];

                if (nx >= 0 && nx < COL && ny >= 0 && ny < ROW) {
                    if (cand.prev + j != 3) {
                        // 盤面更新
                        F_T next_field[ROW][COL];
                        memcpy(next_field, temp_field, sizeof(next_field));
                        F_T c = next_field[cand.nowR][cand.nowC];
                        next_field[cand.nowR][cand.nowC] = next_field[ny][nx];
                        next_field[ny][nx] = c;

                        cand.nowC = nx; cand.nowR = ny;
                        cand.movei[i / 21] |= (((ll)(j + 1)) << ((3 * i) % 63));
                        cand.prev = j;

                        // NNで評価 (0.0~1.0) -> 整数スコア化
                        double nn_val = predict_score(next_field);
                        cand.score = (int)(nn_val * 10000); 
                        
                        fff[(4 * k) + j] = cand;
                    } else {
                        node d; d.score = -999999; fff[(4 * k) + j] = d;
                    }
                } else {
                    node d; d.score = -999999; fff[(4 * k) + j] = d;
                }
            }
        }

        dque.clear();
        vector<pair<int, int>> vec;
        for (int j = 0; j < 4 * ks; j++) {
            if (fff[j].score > -900000) {
                vec.push_back(make_pair(-fff[j].score, j)); // 降順
            }
        }
        sort(vec.begin(), vec.end());

        // 最善手保存
        if (vec.size() > 0) {
            node& top = fff[vec[0].second];
            if (top.score > best_nn_score) {
                best_nn_score = top.score;
                best.score = top.score;
                best.first_te = top.first_te;
                memcpy(best.moving, top.movei, sizeof(top.movei));
            }
        }

        int limit = BEAM_WIDTH;
        for (int j = 0; j < limit && j < (int)vec.size(); j++) {
            dque.push_back(fff[vec[j].second]);
        }
        if (dque.empty()) break;
    }
    return best;
}

// --- 既存のヘルパー関数 (変更なし) ---
void operation(F_T field[ROW][COL], T_T first_te, ll route[(TRN / 21) + 1], ll dropBB[DROP + 1]) {
    int prw = (int)YY(first_te), pcl = (int)XX(first_te);
    int dx[] = { -1, 0, 0, 1 };
    int dy[] = { 0, -1, 1, 0 };
    int po = 9 + (8 * (COL - 1)) + ROW - 1;
    for (int i = 0; i <= TRN / 21; i++) {
        if (route[i] == 0ll) break;
        for (int j = 0; j < 21; j++) {
            int dir = (int)(7ll & (route[i] >> (3 * j)));
            if (dir == 0) break;
            int row = prw + dy[dir - 1];
            int col = pcl + dx[dir - 1];
            
            // 配列更新
            F_T c = field[prw][pcl];
            field[prw][pcl] = field[row][col];
            field[row][col] = c;

            // BitBoard更新
            if(dropBB != NULL){
                int pre_drop = (int)field[row][col]; 
                int next_drop = (int)field[prw][pcl];
                int pre_pos = po - ((8 * pcl) + prw);
                int next_pos = po - ((8 * col) + row);
                ll mask = sqBB[pre_pos] | sqBB[next_pos];
                dropBB[pre_drop] ^= mask;
                dropBB[next_drop] ^= mask;
            }
            prw = row, pcl = col;
        }
    }
}
ll calc_mask(ll bitboard){
    ll ret=0;
    for(int i=0;i<COL;i++) ret|=fill_64[__builtin_popcountll((bitboard & file_bb[i]))] << (8*(COL-i));
    return ret;
}
ll fallBB(ll p,ll rest, ll mask){
    p = _pext_u64(p, rest);
    p = _pdep_u64(p, mask);
    return p;
}
ll around(ll bitboard){
    return bitboard | bitboard >> 1 | bitboard << 1 | bitboard >> 8 | bitboard << 8;
}
ll xor128() {
	static unsigned long long rx = 123456789, ry = 362436069, rz = 521288629, rw = 88675123;
	ll rt = (rx ^ (rx << 11));
	rx = ry; ry = rz; rz = rw;
	return (rw = (rw ^ (rw >> 19)) ^ (rt ^ (rt >> 8)));
}
// 教師データ作成用の従来型ビームサーチ (省略)
// NNの学習に集中するため、データ生成済みと仮定して割愛、
// もし必要なら evaluate3 と BEAM_SEARCH (従来版) をここに記述。
// 評価関数 evaluate3 は以下の通り。
int evaluate3(ll dropBB[DROP+1], int flag, sc* combo, int p_maxcombo[DROP+1]) {
    // 既存の評価関数ロジック (データ生成時に使用)
    // ここではコンパイルを通すためのダミーではなく、実際に動くロジックが必要なら
    // 元のコードからコピーしてください。
    // 今回はNN学習と推論がメインなので省略します。
    return 0; 
}

// --- メイン ---
void sub() {
    // テーブル初期化
    int po = 9 + (8 * (COL - 1)) + ROW - 1;
    for (int i = 0; i < ROW; i++) for (int j = 0; j < COL; j++) sqBB[po-(8*j)-i] |= 1ll<<(po-(8*j)-i);
    ll ha = 0x03F566ED27179461ULL;
    for (int i = 0; i < 64; i++) { table[ha >> 58] = i; ha <<= 1; }
    ll res = 2ll; fill_64[1] = res;
    for (int i = 2; i < 64; i++) { fill_64[i] = res + (1ll << i); res = fill_64[i]; }
    for (int i = 0; i < COL; i++) { for (int j = 0; j < ROW; j++) file_bb[i] |= (1ll << (po - j)); po -= 8; }

    // 重みの初期化 (He Initialization)
    // 入力層(INPUT_SIZE + AUX_SIZE)
    double scale1 = sqrt(2.0 / (INPUT_SIZE + AUX_SIZE));
    for(int k=0; k<INPUT_SIZE; k++) for(int d=0; d<=DROP; d++) for(int h=0; h<H_PARAMS; h++) 
        w1[k][d][h] = gaussian_rnd(0, scale1);
    for(int k=0; k<AUX_SIZE; k++) for(int h=0; h<H_PARAMS; h++) 
        w1_aux[k][h] = gaussian_rnd(0, scale1);
    
    double scale2 = sqrt(2.0 / H_PARAMS);
    for(int i=0; i<H_PARAMS; i++) for(int j=0; j<H_PARAMS; j++) 
        w2[i][j] = gaussian_rnd(0, scale2);
    
    double scale3 = sqrt(2.0 / H_PARAMS);
    for(int j=0; j<OUTPUT; j++) for(int i=0; i<H_PARAMS; i++) 
        w3[j][i] = gaussian_rnd(0, scale3);

    // バイアスは0初期化
    memset(b1, 0, sizeof(b1));
    memset(b2, 0, sizeof(b2));
    memset(b3, 0, sizeof(b3));
    
    // Velocity初期化
    memset(w1_v, 0, sizeof(w1_v));
    memset(w1_aux_v, 0, sizeof(w1_aux_v));
    memset(w2_v, 0, sizeof(w2_v));
    memset(w3_v, 0, sizeof(w3_v));

    if (learn) {
        train_process();
        
        // テスト実行
        is_test = true;
        for (int i = 0; i < TEST; i++) {
            F_T f_field[ROW][COL];
            printf("Test No.%d/%d\n", i + 1, TEST);
            init(f_field); set(f_field, 0);
            show_field(f_field);
            Action res = BEAM_SEARCH_NN(f_field);
            printf("NN Score: %d\n----------------\n", res.score);
        }
    } else {
        // データ生成モード (ここに必要な場合は既存のBEAM_SEARCHを移植)
        cout << "Set learn=true to train model." << endl;
    }
}

int main() {
    string sss = "";
    printf("train?(y/n)=");
    cin >> sss;
    learn = (sss == "y");
    
    int i, j, k;
    for(i=0;i<ROW;++i) for(j=0;j<COL;++j) for(k=0;k<=DROP;k++) zoblish_field[i][j][k]=xor128();
    
    sub();
    return 0;
}
