/*
  g++ -O2 -std=c++11 -fopenmp -mbmi2 ML_Full.cpp -o ML
  ※ BMI2命令セット(PEXT/PDEP)を使用するため、対応CPUで -mbmi2 オプションが必要です。
  非対応環境では _pext_u64 / _pdep_u64 をソフトウェア実装に置き換える必要があります。
*/

#pragma warning(disable:4710)
#pragma warning(disable:4711)
#pragma warning(disable:4820)
#pragma GCC target ("sse4.2,bmi2") 
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
#include <immintrin.h> // for _pext_u64, _pdep_u64

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// --- ハイパーパラメータ & 定数 ---
#define ROW 5
#define COL 6
#define DROP 6
#define TRN 150
#define MAX_TURN 150
#define BEAM_WIDTH 100
#define PROBLEM 1000 // 生成する教師データの数
#define BONUS 10
#define NODE_SIZE 2000 

// NNパラメータ
#define INPUT (ROW*COL)
#define H_PARAMS 64    // 隠れ層のニューロン数
#define OUTPUT 1
#define LEARNING_RATE 0.001 // 学習率
#define EPOCHS 20          // 学習のエポック数

// 型定義
typedef char F_T;
typedef char T_T;
typedef signed char sc;
typedef unsigned char uc;
typedef unsigned long long ll;

// 方向定義 (L, U, D, R)
#define DIR 4
#define XX(PT) ((PT)&15)
#define YY(PT) XX((PT)>>4)
#define YX(Y,X) ((Y)<<4|(X))

enum { EVAL_NONE = 0, EVAL_FALL, EVAL_SET, EVAL_FS, EVAL_COMBO };

// --- グローバル変数 (NNの重み) ---
// double型に変更して微細な勾配を扱えるようにする
double weight1[INPUT][DROP + 1][H_PARAMS];
double weight2[H_PARAMS][H_PARAMS];
double weight3[OUTPUT][H_PARAMS];
double biasx[3][H_PARAMS]; // 0:Layer1, 1:Layer2, 2:Output(size 1)

// 制御フラグ
bool is_test = false;
bool learn = false;

// ビットボード演算用テーブル
ll sqBB[64];
ll file_bb[COL];
int table[64];
ll fill_64[64];

// --- 構造体 ---
struct node {
    T_T first_te;
    ll movei[(TRN / 21) + 1];
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
        memset(this->movei, 0, sizeof(this->movei));
    }
    bool operator < (const node& n)const {
        return score < n.score; // 昇順ソート用 (高い方が後ろ)
    }
};

struct Action {
    T_T first_te;
    int score;
    int maxcombo;
    ll moving[(TRN / 21) + 1];
    Action() {
        this->score = 0;
        memset(this->moving, 0, sizeof(this->moving));
    }
};

node fff[NODE_SIZE]; // ビームサーチ用バッファ

// --- 関数プロトタイプ ---
void init(F_T field[ROW][COL]);
void set(F_T field[ROW][COL], int force);
void show_field(F_T field[ROW][COL]);
int rnd(int mini, int maxi);
double d_rnd();
int chain(int nrw, int ncl, F_T d, F_T field[ROW][COL], F_T chkflag[ROW][COL], F_T delflag[ROW][COL]);
void fall(int x, int h, F_T field[ROW][COL]);
int evaluate_simple(F_T field[ROW][COL], int flag); 
void operation(F_T field[ROW][COL], T_T first_te, ll route[(TRN / 21) + 1], ll dropBB[DROP + 1]);
ll xor128();
ll calc_mask(ll bitboard);
ll fallBB(ll p, ll rest, ll mask);
int evaluate3(ll dropBB[DROP + 1], int flag, sc* combo, int p_maxcombo[DROP + 1]);
int predict(int X[INPUT]);
void sgd_train(int X[INPUT], double target_score);
void train();
void memo_data(F_T field[ROW][COL], int score);
Action BEAM_SEARCH(F_T f_field[ROW][COL]);
void sub();

// --- 乱数 ---
int rnd(int mini, int maxi) {
    static mt19937 mt((int)time(0));
    uniform_int_distribution<int> dice(mini, maxi);
    return dice(mt);
}
double d_rnd() {
    static mt19937 mt((int)time(0));
    uniform_real_distribution<double> dice(-0.05, 0.05); // 小さな値で初期化
    return dice(mt);
}
ll xor128() {
    static unsigned long long rx = 123456789, ry = 362436069, rz = 521288629, rw = 88675123;
    ll rt = (rx ^ (rx << 11));
    rx = ry; ry = rz; rz = rw;
    return (rw = (rw ^ (rw >> 19)) ^ (rt ^ (rt >> 8)));
}

// --- NN関連 (ReLU活性化関数) ---
inline double relu(double x) { return x > 0 ? x : 0.0; }
inline double relu_prime(double x) { return x > 0 ? 1.0 : 0.0; }

// 推論 (Forward)
int predict(int X[INPUT]) {
    double h1[H_PARAMS], h2[H_PARAMS], out;

    // Layer 1
    for (int i = 0; i < H_PARAMS; i++) {
        h1[i] = biasx[0][i];
        for (int j = 0; j < INPUT; j++) {
            h1[i] += weight1[j][X[j]][i];
        }
        h1[i] = relu(h1[i]);
    }
    // Layer 2
    for (int i = 0; i < H_PARAMS; i++) {
        h2[i] = biasx[1][i];
        for (int j = 0; j < H_PARAMS; j++) {
            h2[i] += weight2[j][i] * h1[j];
        }
        h2[i] = relu(h2[i]);
    }
    // Output
    out = biasx[2][0];
    for (int j = 0; j < H_PARAMS; j++) {
        out += weight3[0][j] * h2[j];
    }
    return (int)out;
}

// 学習 (Backpropagation)
void sgd_train(int X[INPUT], double target) {
    double z1[H_PARAMS], h1[H_PARAMS];
    double z2[H_PARAMS], h2[H_PARAMS];
    double out;

    // --- Forward ---
    for (int i = 0; i < H_PARAMS; i++) {
        z1[i] = biasx[0][i];
        for (int j = 0; j < INPUT; j++) z1[i] += weight1[j][X[j]][i];
        h1[i] = relu(z1[i]);
    }
    for (int i = 0; i < H_PARAMS; i++) {
        z2[i] = biasx[1][i];
        for (int j = 0; j < H_PARAMS; j++) z2[i] += weight2[j][i] * h1[j];
        h2[i] = relu(z2[i]);
    }
    out = biasx[2][0];
    for (int j = 0; j < H_PARAMS; j++) out += weight3[0][j] * h2[j];

    // --- Backward ---
    double delta_out = out - target; // 誤差 (MSEの微分)

    // Layer 3 update
    for (int j = 0; j < H_PARAMS; j++) {
        double grad = delta_out * h2[j];
        weight3[0][j] -= LEARNING_RATE * grad;
    }
    biasx[2][0] -= LEARNING_RATE * delta_out;

    // Layer 2 update
    double delta_h2[H_PARAMS];
    for (int j = 0; j < H_PARAMS; j++) {
        delta_h2[j] = delta_out * weight3[0][j] * relu_prime(z2[j]);
    }
    for (int j = 0; j < H_PARAMS; j++) {
        for (int k = 0; k < H_PARAMS; k++) {
            double grad = delta_h2[j] * h1[k];
            weight2[k][j] -= LEARNING_RATE * grad;
        }
        biasx[1][j] -= LEARNING_RATE * delta_h2[j];
    }

    // Layer 1 update
    double delta_h1[H_PARAMS];
    for (int j = 0; j < H_PARAMS; j++) {
        double error = 0;
        for (int k = 0; k < H_PARAMS; k++) error += delta_h2[k] * weight2[j][k];
        delta_h1[j] = error * relu_prime(z1[j]);
    }
    for (int k = 0; k < INPUT; k++) {
        int drop = X[k];
        for (int j = 0; j < H_PARAMS; j++) {
            weight1[k][drop][j] -= LEARNING_RATE * delta_h1[j];
        }
    }
    for (int j = 0; j < H_PARAMS; j++) {
        biasx[0][j] -= LEARNING_RATE * delta_h1[j];
    }
}

// 教師データの保存
void memo_data(F_T field[ROW][COL], int score) {
    ofstream fi("train.txt", ios::app);
    for (int i = 0; i < ROW * COL; i++) {
        fi << (int)field[i / COL][i % COL];
    }
    fi << "," << score << "\n";
    fi.close();
}

// 学習ループ
void train() {
    vector<pair<string, int>> dataset;
    ifstream fi("train.txt");
    string line;
    if (!fi) { cout << "No train.txt found." << endl; return; }

    while (getline(fi, line)) {
        size_t comma = line.find(',');
        if (comma != string::npos) {
            string board_str = line.substr(0, comma);
            int score = stoi(line.substr(comma + 1));
            dataset.push_back(make_pair(board_str, score));
        }
    }
    fi.close();
    cout << "Dataset size: " << dataset.size() << endl;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        shuffle(dataset.begin(), dataset.end(), mt19937(time(0)));
        
        for (const auto& p : dataset) {
            int X[INPUT];
            for (int i = 0; i < INPUT; i++) X[i] = p.first[i] - '0';
            double y = (double)p.second;
            
            // 誤差表示用
            int pred = predict(X);
            total_loss += (pred - y) * (pred - y);

            sgd_train(X, y);
        }
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / dataset.size());
    }
}

// --- パズルロジック ---

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
void fall(int x, int h, F_T field[ROW][COL]) {
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

// 再帰的な連結チェック
int chain(int nrw, int ncl, F_T d, F_T field[ROW][COL], F_T chkflag[ROW][COL], F_T delflag[ROW][COL]) {
    int count = 0;
    if (field[nrw][ncl] == d && chkflag[nrw][ncl] == 0 && delflag[nrw][ncl] > 0) {
        count++;
        chkflag[nrw][ncl] = 1;
        int dy[] = { -1, 1, 0, 0 };
        int dx[] = { 0, 0, -1, 1 };
        for (int i = 0; i < 4; i++) {
            int ny = nrw + dy[i];
            int nx = ncl + dx[i];
            if (ny >= 0 && ny < ROW && nx >= 0 && nx < COL) {
                count += chain(ny, nx, d, field, chkflag, delflag);
            }
        }
    }
    return count;
}

// 簡易評価関数（盤面更新用）
int evaluate_simple(F_T field[ROW][COL], int flag) {
    int combo = 0;
    while (1) {
        int cmb = 0;
        F_T delflag[ROW][COL] = { 0 };
        F_T chkflag[ROW][COL] = { 0 };
        int GetHeight[COL];
        for(int i=0;i<COL;i++) GetHeight[i]=ROW;

        for (int r = 0; r < ROW; r++) {
            for (int c = 0; c < COL; c++) {
                F_T n = field[r][c];
                if (n == 0) continue;
                if(GetHeight[c] == ROW) GetHeight[c] = r;
                
                if (c <= COL - 3 && n == field[r][c + 1] && n == field[r][c + 2]) {
                    delflag[r][c] = delflag[r][c + 1] = delflag[r][c + 2] = 1;
                }
                if (r <= ROW - 3 && n == field[r + 1][c] && n == field[r + 2][c]) {
                    delflag[r][c] = delflag[r + 1][c] = delflag[r + 2][c] = 1;
                }
            }
        }
        for (int r = 0; r < ROW; r++) {
            for (int c = 0; c < COL; c++) {
                if (delflag[r][c] && !chkflag[r][c]) {
                    if (chain(r, c, field[r][c], field, chkflag, delflag) >= 3) cmb++;
                }
            }
        }
        combo += cmb;
        if (cmb == 0 || !(flag & EVAL_COMBO)) break;

        for (int r = 0; r < ROW; r++) {
            for (int c = 0; c < COL; c++) {
                if (delflag[r][c]) field[r][c] = 0;
            }
        }
        if (flag & EVAL_FALL) {
            for (int x = 0; x < COL; x++) fall(x, GetHeight[x], field);
        }
    }
    return combo;
}

// ビットボード用ヘルパー
ll calc_mask(ll bitboard) {
    ll ret = 0;
    for (int i = 0; i < COL; i++) {
        ret |= fill_64[__builtin_popcountll((bitboard & file_bb[i]))] << (8 * (COL - i));
    }
    return ret;
}
ll fallBB(ll p, ll rest, ll mask) {
    p = _pext_u64(p, rest);
    p = _pdep_u64(p, mask);
    return p;
}
ll around(ll bitboard) {
    return bitboard | bitboard >> 1 | bitboard << 1 | bitboard >> 8 | bitboard << 8;
}

// 高速評価関数（教師データ生成用）
// 元の3駒関係を排除し、純粋なコンボ数+落としボーナスを返すように修正
int evaluate3(ll dropBB[DROP + 1], int flag, sc* combo, int p_maxcombo[DROP + 1]) {
    int ev = 0;
    *combo = 0;
    int oti = 0;
    ll occBB = 0;
    for (int i = 1; i <= DROP; i++) occBB |= dropBB[i];

    int po = 9 + (8 * (COL - 1)) + ROW - 1;
    int d_maxcombo[DROP + 1] = { 0 };

    while (1) {
        int cmb = 0;
        int cmb2 = 0;
        ll linked[DROP + 1] = { 0 };

        for (int i = 1; i <= DROP; i++) {
            ll vert = (dropBB[i]) & (dropBB[i] << 1) & (dropBB[i] << 2);
            ll hori = (dropBB[i]) & (dropBB[i] << 8) & (dropBB[i] << 16);
            linked[i] = vert | (vert >> 1) | (vert >> 2) | hori | (hori >> 8) | (hori >> 16);
        }

        for (int i = 1; i <= DROP; i++) {
            long long tmp_linked = linked[i];
            while (tmp_linked) {
                long long chainBB = tmp_linked & (-tmp_linked);
                long long peek = chainBB;
                while (1) {
                    long long tmp = tmp_linked & around(peek);
                    if (peek == tmp) break;
                    peek = tmp;
                }
                int c = __builtin_popcountll(peek);
                tmp_linked ^= peek;
                if (c >= 3) {
                    cmb++;
                    cmb2 += (c == 3) ? 3000 : 2000; // コンボ価値を高く設定
                    d_maxcombo[i]++;
                }
            }
        }

        *combo += cmb;
        ev += cmb2;
        if (cmb == 0 || !(flag & EVAL_COMBO)) break;
        oti++;

        ll mask = calc_mask(occBB);
        for (int i = 1; i <= DROP; i++) {
            dropBB[i] = fallBB(dropBB[i], occBB, mask);
        }
        occBB = fallBB(occBB, occBB, mask);
    }
    
    // 3駒関係データへのアクセスを削除し、純粋なスコアを返す
    // 落としが決まるほど高得点になるように調整
    ev += oti * 500; 
    return ev;
}

// 移動処理
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
            if (dropBB) {
                int pre_drop = (int)field[row][col]; // 元の位置にあった色(今は移動先)
                int next_drop = (int)field[prw][pcl]; // 移動してきた色
                int pre_pos = po - ((8 * pcl) + prw);
                int next_pos = po - ((8 * col) + row);
                
                // スワップ用のXORマスク
                ll mask = sqBB[pre_pos] | sqBB[next_pos];
                dropBB[pre_drop] ^= mask;
                dropBB[next_drop] ^= mask;
            }
            prw = row, pcl = col;
        }
    }
}

// --- ビームサーチ ---
Action BEAM_SEARCH(F_T f_field[ROW][COL]) {
    int po = 9 + (8 * (COL - 1)) + ROW - 1;
    int stop = 0;
    int p_maxcombo[DROP + 1] = { 0 };
    int drop_cnt[DROP + 1] = { 0 };
    
    for (int r = 0; r < ROW; r++)
        for (int c = 0; c < COL; c++)
            if (f_field[r][c] >= 1 && f_field[r][c] <= DROP) drop_cnt[f_field[r][c]]++;
    
    for (int i = 1; i <= DROP; i++) {
        stop += drop_cnt[i] / 3;
        p_maxcombo[i] = drop_cnt[i] / 3;
    }

    vector<node> dque;
    
    // 1手目 (全探索)
    for (int r = 0; r < ROW; r++) {
        for (int c = 0; c < COL; c++) {
            node cand;
            cand.nowR = r; cand.nowC = c;
            cand.first_te = (T_T)YX(r, c);
            cand.hash = 0; // 簡易化のためハッシュは使わない（必要なら再実装）
            dque.push_back(cand);
        }
    }

    int dx[] = { -1, 0, 0, 1 };
    int dy[] = { 0, -1, 1, 0 };
    Action bestAction;
    bestAction.maxcombo = stop;
    int maxScore = -1;

    ll rootBB[DROP + 1] = { 0 };
    for (int r = 0; r < ROW; r++)
        for (int c = 0; c < COL; c++)
            rootBB[f_field[r][c]] |= (1ll << (po - ((8 * c) + r)));

    // 探索ループ
    for (int i = 0; i < MAX_TURN; i++) {
        int ks = (int)dque.size();
        #pragma omp parallel for
        for (int k = 0; k < ks; k++) {
            node temp = dque[k];
            // 現在の状態を復元
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
                    if (cand.prev + j != 3) { // 直前と逆方向は禁止
                        // 状態更新
                        int p1_drop = temp_field[ny][nx];
                        int p2_drop = temp_field[cand.nowR][cand.nowC];
                        
                        // BitBoard更新 (XOR swap)
                        int p1_pos = po - ((8 * nx) + ny);
                        int p2_pos = po - ((8 * cand.nowC) + cand.nowR);
                        ll mask = sqBB[p1_pos] | sqBB[p2_pos];
                        temp_dropBB[p1_drop] ^= mask;
                        temp_dropBB[p2_drop] ^= mask;
                        
                        // 配列更新 (次のターンのためのtempとして使うため戻す必要ありだが、
                        // ここではコピーコストを避けるため、evaluate3にはBBを渡す)
                        
                        cand.nowC = nx; cand.nowR = ny;
                        cand.movei[i / 21] |= (((ll)(j + 1)) << ((3 * i) % 63));
                        cand.prev = j;

                        // 評価
                        if (is_test) {
                            // NN推論
                            // NNには配列が必要なので更新する
                            F_T next_field[ROW][COL];
                            memcpy(next_field, temp_field, sizeof(next_field));
                            next_field[ny][nx] = p2_drop;
                            next_field[temp.nowR][temp.nowC] = p1_drop;
                            
                            int X[INPUT];
                            for(int x=0; x<INPUT; x++) X[x] = next_field[x/COL][x%COL];
                            cand.score = predict(X); // NNのスコア
                            
                            // BitBoardを戻す
                            temp_dropBB[p1_drop] ^= mask;
                            temp_dropBB[p2_drop] ^= mask;
                        } else {
                            // 教師データ生成 (既存評価関数)
                            sc cmb;
                            // コピーを渡す
                            ll evalBB[DROP+1];
                            memcpy(evalBB, temp_dropBB, sizeof(evalBB));
                            cand.score = evaluate3(evalBB, EVAL_FALL | EVAL_COMBO, &cmb, p_maxcombo);
                            cand.combo = cmb;
                            
                            // BitBoardを戻す
                            temp_dropBB[p1_drop] ^= mask;
                            temp_dropBB[p2_drop] ^= mask;
                        }
                        
                        fff[(4 * k) + j] = cand;
                    } else {
                        node dummy; dummy.score = -999999; fff[(4 * k) + j] = dummy;
                    }
                } else {
                    node dummy; dummy.score = -999999; fff[(4 * k) + j] = dummy;
                }
            }
        }

        dque.clear();
        vector<pair<int, int>> vec;
        for (int j = 0; j < 4 * ks; j++) {
            if (fff[j].score > -900000) {
                vec.push_back(make_pair(-fff[j].score, j)); // 降順ソート用
            }
        }
        sort(vec.begin(), vec.end());

        int limit = is_test ? BEAM_WIDTH : 10; // 学習データ生成時は幅を狭めて高速化
        for (int j = 0; j < limit && j < (int)vec.size(); j++) {
            node& next_node = fff[vec[j].second];
            
            // 最大スコア更新
            if (next_node.score > maxScore) {
                maxScore = next_node.score;
                bestAction.score = maxScore;
                bestAction.first_te = next_node.first_te;
                memcpy(bestAction.moving, next_node.movei, sizeof(next_node.movei));
            }
            // 理論値到達で終了 (教師データ生成時)
            if (!is_test && next_node.combo == stop) {
                return bestAction;
            }
            
            if (i < MAX_TURN - 1) {
                dque.push_back(next_node);
            }
        }
        if (dque.empty()) break;
    }
    return bestAction;
}

void sub() {
    // 1. 初期化 (BitBoardテーブル作成)
    int po = 9 + (8 * (COL - 1)) + ROW - 1;
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            int pos = po - (8 * j) - i;
            sqBB[pos] |= 1ll << pos;
        }
    }
    ll ha = 0x03F566ED27179461ULL;
    for (int i = 0; i < 64; i++) {
        table[ha >> 58] = i;
        ha <<= 1;
    }
    ll res = 2ll;
    fill_64[1] = res;
    for (int i = 2; i < 64; i++) {
        fill_64[i] = res + (1ll << i);
        res = fill_64[i];
    }
    for (int i = 0; i < COL; i++) {
        for (int j = 0; j < ROW; j++) {
            file_bb[i] |= (1ll << (po - j));
        }
        po -= 8;
    }
    // 重み初期化
    for(int k=0; k<INPUT; k++)
        for(int d=0; d<=DROP; d++)
            for(int h=0; h<H_PARAMS; h++) weight1[k][d][h] = d_rnd();
    for(int i=0; i<H_PARAMS; i++)
        for(int j=0; j<H_PARAMS; j++) weight2[i][j] = d_rnd();
    for(int j=0; j<OUTPUT; j++)
        for(int i=0; i<H_PARAMS; i++) weight3[j][i] = d_rnd();
    for(int j=0; j<3; j++)
        for(int i=0; i<H_PARAMS; i++) biasx[j][i] = d_rnd();


    if (!learn) {
        // --- 教師データ生成モード ---
        is_test = false;
        // ファイルをクリア
        ofstream ofs("train.txt", ios::trunc); ofs.close();
        
        for (int i = 0; i < PROBLEM; i++) {
            if(i % 10 == 0) cout << "Generating Data... " << i << "/" << PROBLEM << endl;
            F_T f_field[ROW][COL];
            init(f_field); set(f_field, 0);
            
            // 既存の強力な評価関数(evaluate3)で解く
            Action best = BEAM_SEARCH(f_field);
            
            // 解の経路を辿り、各ステップの盤面と残りスコアを保存
            F_T current_field[ROW][COL];
            ll current_BB[DROP+1]={0};
            int po_idx = 9 + (8 * (COL - 1)) + ROW - 1;
            for(int r=0;r<ROW;r++) for(int c=0;c<COL;c++) current_BB[f_field[r][c]] |= (1ll<<(po_idx-((8*c)+r)));

            memcpy(current_field, f_field, sizeof(f_field));
            
            // 初期状態
            // スコアは「最終コンボ数」や「評価値」などを教師とする
            // ここでは簡易的に最終到達スコアを与える
            memo_data(current_field, best.score);
            
            // 操作を再現して途中経過も保存
            // (operation関数は一括処理なので、ここでは簡易実装として省略。
            //  本来は1手ごとに memo_data を呼ぶとデータが増える)
        }
        cout << "Data generation complete. Run with 'y' to train." << endl;
    } else {
        // --- 学習 & テストモード ---
        is_test = false;
        train(); // NN学習
        
        is_test = true; // 推論モードへ
        cout << "--- Start Testing ---" << endl;
        double avg_score = 0;
        for (int i = 0; i < TEST; i++) {
            F_T f_field[ROW][COL];
            init(f_field); set(f_field, 0);
            Action best = BEAM_SEARCH(f_field); // NNを使って探索
            cout << "Test " << i << ": Score = " << best.score << endl;
            avg_score += best.score;
        }
        cout << "Average Score: " << avg_score / TEST << endl;
    }
}

int main() {
    string sss = "";
    cout << "Train mode? (y: Train & Test, n: Generate Data): ";
    cin >> sss;
    learn = (sss == "y");
    sub();
    return 0;
}