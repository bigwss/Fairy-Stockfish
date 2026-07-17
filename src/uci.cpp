/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "evaluate.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "xboard.h"
#include "syzygy/tbprobe.h"
#include "nlohmann/json.hpp"

using namespace std;

namespace Stockfish
{

extern vector<string> setup_bench(const Position &, istream &);

namespace
{

struct MoveRecord {
    int human_index = 0;            // 第几个人类走子(1-based)
    int ply_index = 0;              // 全局半回合序号
    std::string fen_before;         // 人类落子前 FEN
    std::string move;               // 人类实走(UCI)
    std::string best_move;          // 引擎推理的最佳走法(UCI)
    std::optional<double> best_cp;  // 裁判填:最佳分(人类视角)
    std::optional<double> best_win_rate;
    std::optional<double> played_cp;        // 裁判填:实走分(人类视角)
    std::optional<double> played_win_rate;  // 裁判填:实走分(人类视角)
    std::optional<double> loss;             // 裁判填:胜率损失
};

struct MoveRecord2 {
    int human_index = 0;            // 第几个人类走子(1-based)
    int ply_index = 0;              // 全局半回合序号
    std::string fen_before;         // 人类落子前 FEN
    std::string move;               // 人类实走(UCI)
    std::string best_move;          // 引擎推理的最佳走法(UCI)
    std::optional<double> best_cp;  // 裁判填:最佳分(人类视角)
    std::optional<double> best_win_rate;
    std::optional<double> played_cp;        // 裁判填:实走分(人类视角)
    std::optional<double> played_win_rate;  // 裁判填:实走分(人类视角)
    std::optional<double> loss;             // 裁判填:胜率损失

    // 强裁判在“人类/被测等级走子前局面”下搜索出的候选着法列表。
    // 由 MultiPV 200 得到，按裁判分数从好到坏排序。每个元素包含：
    //   rank: 候选排名，1 表示最佳着
    //   move: 候选着法(UCI/ICCS 坐标)
    //   cp: 候选分数，单位 centipawn，方向与当前走子方一致
    //   win_rate: 候选胜率，单位百分比，例如 5.3 表示 5.3%
    //   loss_from_best: 与最佳候选的胜率差，单位百分点
    nlohmann::json oracle_candidates = nlohmann::json::array();

    // oracle_candidates 中实际保存的候选数量。正常情况下等于当前局面的合法着数；
    // 如果合法着数超过 MultiPV 上限，或引擎没有完整返回，这个值可用于发现截断。
    int candidate_count = 0;

    // 实走着法在 oracle_candidates 中的排名，1 表示最佳着。
    // -1 表示未找到实走，一般说明输入走法异常或候选列表被截断。
    int played_rank = -1;

    // 比实走着法排名更靠前的候选数量。若 played_rank=1，则为 0。
    // -1 表示实走未找到。
    int num_better_moves = -1;

    // 实走是否命中前 N 名候选。用于粗粒度描述“是否在强裁判可接受范围内”。
    bool top1_hit = false;
    bool top3_hit = false;
    bool top5_hit = false;

    // 实走在候选排序中的百分位，范围 [0,1]，越大越好。
    // rank=1 时为 1；最后一名约为 0。实走未找到时不输出。
    std::optional<double> played_score_percentile;

    // 最佳着与第二佳着的 cp 差。值越大，说明第一选择越唯一；
    // 这种局面通常比“很多着法都差不多”的局面更有区分度。
    std::optional<double> best_second_cp_gap;

    // 最佳着与候选 cp 中位数的差。值越大，说明普通着法和最佳着差距更大；
    // 可作为局面“考点强度/候选分化程度”的一个摘要。
    std::optional<double> best_median_cp_gap;

    // 所有候选 cp 的标准差。值越大，说明候选着法质量分布越分散；
    // 候选分布越分散，玩家/等级的选择越可能带来可观测差异。
    std::optional<double> candidate_cp_std;

    // 与最佳着 cp 差不超过 50/100 的候选数量，称为“安全着数量”。
    // safe_count 越大，说明好走法很多；只看 loss 时这种局面容易把等级差异抹平。
    int safe_count_50cp = 0;
    int safe_count_100cp = 0;

    // 与最佳着 cp 差至少 150/300 的候选数量，称为“坏着数量”。
    // bad_count 越大，说明局面里可犯错空间更大。
    int bad_count_150cp = 0;
    int bad_count_300cp = 0;
};

static nlohmann::json move_record_to_json(const MoveRecord2 &r)
{
    nlohmann::json j;
    j["human_index"] = r.human_index;
    j["ply_index"] = r.ply_index;
    j["fen_before"] = r.fen_before;
    j["move"] = r.move;
    j["best_move"] = r.best_move;
    if (r.best_cp)
        j["best_cp"] = *r.best_cp;
    if (r.played_cp)
        j["played_cp"] = *r.played_cp;
    if (r.loss)
        j["loss"] = *r.loss;
    if (r.best_win_rate)
        j["best_win_rate"] = *r.best_win_rate;
    if (r.played_win_rate)
        j["played_win_rate"] = *r.played_win_rate;
    j["oracle_candidates"] = r.oracle_candidates;
    j["candidate_count"] = r.candidate_count;
    j["played_rank"] = r.played_rank;
    j["num_better_moves"] = r.num_better_moves;
    j["top1_hit"] = r.top1_hit;
    j["top3_hit"] = r.top3_hit;
    j["top5_hit"] = r.top5_hit;
    if (r.played_score_percentile)
        j["played_score_percentile"] = *r.played_score_percentile;
    if (r.best_second_cp_gap)
        j["best_second_cp_gap"] = *r.best_second_cp_gap;
    if (r.best_median_cp_gap)
        j["best_median_cp_gap"] = *r.best_median_cp_gap;
    if (r.candidate_cp_std)
        j["candidate_cp_std"] = *r.candidate_cp_std;
    j["safe_count_50cp"] = r.safe_count_50cp;
    j["safe_count_100cp"] = r.safe_count_100cp;
    j["bad_count_150cp"] = r.bad_count_150cp;
    j["bad_count_300cp"] = r.bad_count_300cp;
    return j;
}

static nlohmann::json move_record_to_json(const MoveRecord &r)
{
    nlohmann::json j;
    j["human_index"] = r.human_index;
    j["ply_index"] = r.ply_index;
    j["fen_before"] = r.fen_before;
    j["move"] = r.move;
    j["best_move"] = r.best_move;
    if (r.best_cp)
        j["best_cp"] = *r.best_cp;
    if (r.played_cp)
        j["played_cp"] = *r.played_cp;
    if (r.loss)
        j["loss"] = *r.loss;
    if (r.best_win_rate)
        j["best_win_rate"] = *r.best_win_rate;
    if (r.played_win_rate)
        j["played_win_rate"] = *r.played_win_rate;
    return j;
}

// 计算一组 cp 分数的总体标准差。
// 输入 xs: 同一局面下所有候选着法的 cp 分数，方向与当前走子方一致。
// 返回值: cp 标准差。越大表示候选着法质量差异越大。
static double cp_stddev(const std::vector<double> &xs)
{
    if (xs.empty())
        return 0.0;

    double sum = 0.0;
    for (double x : xs)
        sum += x;
    const double mean = sum / xs.size();

    double var = 0.0;
    for (double x : xs) {
        const double d = x - mean;
        var += d * d;
    }
    return std::sqrt(var / xs.size());
}

// 将一个 RootMove 候选转为 JSON。
// pos: 当前待分析局面，用于把内部 Move 转成 UCI/ICCS 字符串。
// rm: 搜索得到的一条根候选。
// rank: rm 在 root_moves 中的 1-based 排名。
// best_win: 最佳候选的 win 值，单位是千分制。用于计算 loss_from_best。
static nlohmann::json root_move_to_json(const Position &pos, const Search::RootMove &rm, int rank,
                                        int best_win)
{
    nlohmann::json j;
    const Value v = rm.score;
    std::tuple<int, int, int> wv;
    UCI::wdl(v, pos.game_ply(), wv);
    const int win = std::get<0>(wv);

    j["rank"] = rank;
    j["move"] = rm.pv.empty() ? std::string() : UCI::move(pos, rm.pv[0]);
    j["cp"] = UCI::value_cp(v);
    j["win_rate"] = win / 10.0;
    j["loss_from_best"] = std::max(0.0, (best_win - win) / 10.0);
    return j;
}

// 从强裁判 MultiPV 结果中填充 MoveRecord2 的候选结构特征。
//
// rec:
//   当前被测走子的记录。进入函数前需要已填好 rec.move，即被测方真实走法。
// pos:
//   被测方走子前的局面。所有 root_moves 的分数都以该局面走子方为视角。
// root_moves:
//   go depth N + MultiPV 后得到的根候选，Stockfish 已按分数从好到坏排序。
//
// 本函数做三类事情：
//   1. 保存 oracle_candidates: 每个候选的 rank/move/cp/win_rate/loss_from_best。
//   2. 找到 rec.move 的排名，填充 played_rank、topN_hit、num_better_moves。
//   3. 计算候选集合摘要，如 best-second gap、best-median gap、cp 标准差、
//      safe_count/bad_count，用来描述“这个局面有多能区分玩家选择”。
static void fill_oracle_candidate_features(MoveRecord2 &rec, const Position &pos,
                                           const Search::RootMoves &root_moves)
{
    rec.candidate_count = int(root_moves.size());
    if (root_moves.empty())
        return;

    std::tuple<int, int, int> best_wv;
    UCI::wdl(root_moves[0].score, pos.game_ply(), best_wv);
    // best_win: 最佳候选的胜率，千分制整数。例如 53 表示 5.3%。
    // best_cp: 最佳候选的 cp 分数，后续所有 safe/bad 阈值都相对它计算。
    const int best_win = std::get<0>(best_wv);
    const double best_cp = UCI::value_cp(root_moves[0].score);

    // 保存所有候选 cp，用于计算候选分布的中位数和标准差。
    std::vector<double> cps;
    cps.reserve(root_moves.size());

    for (std::size_t i = 0; i < root_moves.size(); ++i) {
        const auto &rm = root_moves[i];
        if (rm.pv.empty())
            continue;

        const int rank = int(i) + 1;
        const std::string move = UCI::move(pos, rm.pv[0]);
        const double cp = UCI::value_cp(rm.score);
        cps.push_back(cp);

        rec.oracle_candidates.push_back(root_move_to_json(pos, rm, rank, best_win));

        // cp_loss: 当前候选相比最佳候选少多少 cp。
        // 阈值都以 cp_loss 为准：safe 表示接近最佳，bad 表示明显差。
        const double cp_loss = best_cp - cp;
        if (cp_loss <= 50.0)
            rec.safe_count_50cp++;
        if (cp_loss <= 100.0)
            rec.safe_count_100cp++;
        if (cp_loss >= 150.0)
            rec.bad_count_150cp++;
        if (cp_loss >= 300.0)
            rec.bad_count_300cp++;

        if (move == rec.move && rec.played_rank < 0) {
            rec.played_rank = rank;
            rec.num_better_moves = rank - 1;
            rec.top1_hit = rank == 1;
            rec.top3_hit = rank <= 3;
            rec.top5_hit = rank <= 5;
            if (root_moves.size() == 1)
                rec.played_score_percentile = 1.0;
            else
                // 百分位只依赖排序，不依赖 cp 尺度；越接近 1 表示越靠前。
                rec.played_score_percentile =
                    1.0 - double(rank - 1) / double(root_moves.size() - 1);
        }
    }

    if (root_moves.size() >= 2)
        // best-second gap: 最佳候选和第二候选的 cp 差，用来衡量最佳着是否唯一。
        rec.best_second_cp_gap =
            UCI::value_cp(root_moves[0].score) - UCI::value_cp(root_moves[1].score);

    if (!cps.empty()) {
        std::vector<double> sorted = cps;
        std::sort(sorted.begin(), sorted.end());
        // median 是候选集合的“普通着法”代表，best-median gap 越大说明局面越尖锐。
        const double median =
            sorted.size() % 2 == 1
                ? sorted[sorted.size() / 2]
                : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
        rec.best_median_cp_gap = best_cp - median;
        rec.candidate_cp_std = cp_stddev(cps);
    }
}

// position() is called when engine receives the "position" UCI command.
// The function sets up the position described in the given FEN string ("fen")
// or the starting position ("startpos") and then makes the moves given in the
// following move list ("moves").

void position(Position &pos, istringstream &is, StateListPtr &states)
{

    Move m;
    string token, fen;

    is >> token;
    // Parse as SFEN if specified
    bool sfen = token == "sfen";

    if (token == "startpos") {
        fen = variants.find(Options["UCI_Variant"])->second->startFen;
        is >> token;  // Consume "moves" token if any
    } else if (token == "fen" || token == "sfen")
        while (is >> token && token != "moves")
            fen += token + " ";
    else
        return;

    states = StateListPtr(new std::deque<StateInfo>(1));  // Drop old and create a new one
    pos.set(variants.find(Options["UCI_Variant"])->second, fen, Options["UCI_Chess960"],
            &states->back(), Threads.main(), sfen);

    // Parse move list (if any)
    while (is >> token && (m = UCI::to_move(pos, token)) != MOVE_NONE) {
        states->emplace_back();
        pos.do_move(m, states->back());
    }
}

// trace_eval() prints the evaluation for the current position, consistent with the UCI
// options set so far.

void trace_eval(Position &pos)
{

    StateListPtr states(new std::deque<StateInfo>(1));
    Position p;
    p.set(pos.variant(), pos.fen(), Options["UCI_Chess960"], &states->back(), Threads.main());

    Eval::NNUE::verify();

    sync_cout << "\n" << Eval::trace(p) << sync_endl;
}

// setoption() is called when engine receives the "setoption" UCI command. The
// function updates the UCI option ("name") to the given value ("value").

void setoption(istringstream &is)
{

    string token, name, value;

    is >> token;  // Consume "name" token

    if (Options["Protocol"] == "ucci")
        name = token;
    else
        // Read option name (can contain spaces)
        while (is >> token && token != "value")
            name += (name.empty() ? "" : " ") + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;

    if (Options.count(name))
        Options[name] = value;
    // UCI dialects do not allow spaces
    else if ((Options["Protocol"] == "ucci" || Options["Protocol"] == "usi") &&
             (std::replace(name.begin(), name.end(), '_', ' '), Options.count(name)))
        Options[name] = value;
    else
        sync_cout << "No such option: " << name << sync_endl;
}

// go() is called when engine receives the "go" UCI command. The function sets
// the thinking time and other parameters from the input string, then starts
// the search.

void go(Position &pos, istringstream &is, StateListPtr &states,
        const std::vector<Move> &banmoves = {})
{
    Search::LimitsType limits;
    string token;
    bool ponderMode = false;

    limits.startTime = now();  // As early as possible!

    limits.banmoves = banmoves;
    bool isUsi = Options["Protocol"] == "usi";

    while (is >> token)
        if (token == "searchmoves")  // Needs to be the last command on the line
            while (is >> token)
                limits.searchmoves.push_back(UCI::to_move(pos, token));

        else if (token == "wtime")
            is >> limits.time[isUsi ? BLACK : WHITE];
        else if (token == "btime")
            is >> limits.time[isUsi ? WHITE : BLACK];
        else if (token == "winc")
            is >> limits.inc[isUsi ? BLACK : WHITE];
        else if (token == "binc")
            is >> limits.inc[isUsi ? WHITE : BLACK];
        else if (token == "movestogo")
            is >> limits.movestogo;
        else if (token == "depth")
            is >> limits.depth;
        else if (token == "nodes")
            is >> limits.nodes;
        else if (token == "movetime")
            is >> limits.movetime;
        else if (token == "mate")
            is >> limits.mate;
        else if (token == "perft")
            is >> limits.perft;
        else if (token == "infinite")
            limits.infinite = 1;
        else if (token == "ponder")
            ponderMode = true;
        // UCCI commands
        else if (token == "time")
            is >> limits.time[pos.side_to_move()];
        else if (token == "opptime")
            is >> limits.time[~pos.side_to_move()];
        else if (token == "increment")
            is >> limits.inc[pos.side_to_move()];
        else if (token == "oppinc")
            is >> limits.inc[~pos.side_to_move()];
        // USI commands
        else if (token == "byoyomi") {
            int byoyomi = 0;
            is >> byoyomi;
            limits.inc[WHITE] = limits.inc[BLACK] = byoyomi;
            limits.time[WHITE] += byoyomi;
            limits.time[BLACK] += byoyomi;
        }

    Threads.start_thinking(pos, states, limits, ponderMode);
}

/**
 * @brief
 * @note 评估人走棋的胜率损失，使用AI的最优走法作为最优走法，计算最优走法和is中人的实际走法胜率差
 * @param  is: move record
 * example: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w moves b2e2 b9c7 c3c4
 * a9b9 g3g4 b7a7 b0c2 g6g5 g4g5 b9b5 g5g6 g9e7 h2f2 h9f8
 * @param  human_play_red true 人走的红方 false： 人走的黑方
 * @retval None
 */
void estimate(std::istringstream &is, bool human_play_red)
{
    // 将分析结果保存在这个records中
    std::vector<MoveRecord> records;

    Search::clear();
    Position pos;
    StateListPtr states(new std::deque<StateInfo>(1));
    assert(variants.find(Options["UCI_Variant"])->second != nullptr);
    pos.set(variants.find(Options["UCI_Variant"])->second,
            variants.find(Options["UCI_Variant"])->second->startFen, false, &states->back(),
            Threads.main());
    std::string token("ucci");
    Options["Protocol"].set_default(token);
    string defaultVariant = string(
#ifdef LARGEBOARDS
        token == "usi"    ? "shogi"
        : token == "ucci" ? "xiangqi"
#else
        token == "usi"    ? "minishogi"
        : token == "ucci" ? "minixiangqi"
#endif
                          : "chess");
    Options["UCI_Variant"].set_default(defaultVariant);
    std::istringstream ss("UCI_ShowWDL true");
    setoption(ss);
    ss.clear();
    ss = std::istringstream("EvalFile xiangqi-c07e94a5c7cb.nnue");
    setoption(ss);

    std::deque<std::string> moves;
    std::string fen;

    while (is >> token) {
        std::cout << "token: " << token << std::endl;
        if (token == "moves") {
            while (is >> token) {
                moves.push_back(token);
            }
        } else {
            fen += token + " ";
        }
    }

    bool first_red = true;
    // 根据fen知道谁先走
    auto iter = fen.find('w');
    if (iter == std::string::npos) {
        first_red = false;
    }
    std::cout << "fen: " << fen << "first " << (first_red ? "red" : "black") << std::endl;

    int start_pos = (human_play_red == first_red ? 0 : 1);

    for (std::size_t index = start_pos; index < moves.size(); index += 2) {
        states = StateListPtr(new std::deque<StateInfo>(1));
        pos.set(variants.find(Options["UCI_Variant"])->second, fen, Options["UCI_Chess960"],
                &states->back(), Threads.main(), false);
        std::size_t i = 0;
        for (; i < index; ++i) {
            states->emplace_back();
            pos.do_move(UCI::to_move(pos, moves[i]), states->back());
        }
        // 这里应该判断下是否moves已经消耗完了，break出去
        if (i >= moves.size())
            break;
        sync_cout << pos << sync_endl;
        std::istringstream gois("go depth 14");
        // 最优走法
        go(pos, gois, states);
        Threads.main()->wait_for_search_finished();
        auto bestThread = Threads.get_best_thread();
        std::string best_move = UCI::move(pos, bestThread->rootMoves[0].pv[0]);
        Value v = bestThread->rootMoves[0].score;
        std::tuple<int, int, int> wv;
        UCI::wdl(v, pos.game_ply(), wv);
        const int bestWin = std::get<0>(wv);
        const double bestCp = UCI::value_cp(v);

        double playedCp(bestCp);
        int playedWin(bestWin);
        // 走法不是最优
        if (best_move != moves[index]) {
            // go depth 10 searchmoves xx 设置实际走法
            std::string playedCmd = "go depth 14 searchmoves " + moves[index];
            std::istringstream playedIs(playedCmd);
            go(pos, playedIs, states);
            Threads.main()->wait_for_search_finished();
            bestThread = Threads.get_best_thread();
            Value playedValue = bestThread->rootMoves[0].score;
            std::tuple<int, int, int> playedWv;
            UCI::wdl(playedValue, pos.game_ply(), playedWv);
            playedWin = std::get<0>(playedWv);
            playedCp = UCI::value_cp(playedValue);
        }

        MoveRecord rec;
        rec.human_index = int((index - start_pos) / 2 + 1);
        rec.ply_index = int(index + 1);
        rec.fen_before = pos.fen();
        rec.move = moves[index];
        rec.best_move = best_move;
        rec.best_cp = bestCp;
        rec.played_cp = playedCp;
        rec.loss = (bestWin - playedWin) / 10.0;
        rec.best_win_rate = bestWin / 10.0;
        rec.played_win_rate = playedWin / 10.0;
        // 再次推理由于缓存的存在，可能推导出更优的走法或者导致结果变化，如果推理出用户走法更优，需要保存loss>=0;
        if (rec.loss < 0) {
            rec.loss = 0.0;
        }

        records.push_back(rec);
    }

    nlohmann::json recs = nlohmann::json::array();
    for (const auto &r : records)
        recs.push_back(move_record_to_json(r));
    nlohmann::json j;
    j["records"] = std::move(recs);
    sync_cout << "estimateresult " << j.dump() << sync_endl;
#if 0
    for (const auto &rec : records) {
        std::cout << "estimate human " << rec.human_index << " ply " << rec.ply_index << " fen "
                  << rec.fen_before << " move " << rec.move << " best move: " << rec.best_move;
        if (rec.best_cp.has_value())
            std::cout << " best_cp " << *rec.best_cp;
        if (rec.played_cp.has_value())
            std::cout << " played_cp " << *rec.played_cp;
        if (rec.loss.has_value())
            std::cout << " loss " << *rec.loss;
        std::cout << std::endl;
    }
#endif
}

void estimate2(std::istringstream &is, bool human_play_red)
{
    // 解析形如 "<FEN> moves <move1> <move2> ..." 的输入，逐回合评估人类一方的实走。
    // 与 estimate() 不同：本函数通过一次 MultiPV 搜索同时获得最优着和实走的评分。
    std::vector<MoveRecord2> records;

    // 建立象棋变体的初始局面，并设置本次分析依赖的协议、变体和 NNUE 网络。
    Search::clear();
    Position pos;
    StateListPtr states(new std::deque<StateInfo>(1));
    assert(variants.find(Options["UCI_Variant"])->second != nullptr);
    pos.set(variants.find(Options["UCI_Variant"])->second,
            variants.find(Options["UCI_Variant"])->second->startFen, false, &states->back(),
            Threads.main());
    std::string token("ucci");
    Options["Protocol"].set_default(token);
    string defaultVariant = string(
#ifdef LARGEBOARDS
        token == "usi"    ? "shogi"
        : token == "ucci" ? "xiangqi"
#else
        token == "usi"    ? "minishogi"
        : token == "ucci" ? "minixiangqi"
#endif
                          : "chess");
    Options["UCI_Variant"].set_default(defaultVariant);
    std::istringstream ss("UCI_ShowWDL true");
    setoption(ss);
    ss.clear();
    ss = std::istringstream("EvalFile xiangqi-c07e94a5c7cb.nnue");
    setoption(ss);

    std::deque<std::string> moves;
    std::string fen;

    // "moves" 之前的 token 组成 FEN；其后的全部 token 都是按 ply 顺序排列的走法。
    while (is >> token) {
        // std::cout << "token: " << token << std::endl;
        if (token == "moves") {
            while (is >> token) {
                moves.push_back(token);
            }
        } else {
            fen += token + " ";
        }
    }

    // FEN 中的 "w" 表示红方（先手）走；据此确定走法序列中哪一侧属于人类。
    bool first_red = true;
    auto iter = fen.find('w');
    if (iter == std::string::npos) {
        first_red = false;
    }
    std::cout << "fen: " << fen << "first " << (first_red ? "red" : "black") << std::endl;

    // moves[0] 属于 FEN 指定的行棋方：人类先走时从下标 0 开始，否则从下标 1 开始。
    int start_pos = (human_play_red == first_red ? 0 : 1);

    auto setcmd = [](const std::string &cmd) {
        istringstream iss(cmd);
        setoption(iss);
    };
    // 临时扩大 MultiPV，以便候选中尽可能包含所有合法着及人类的实走；结束后恢复原设置。
    const int oldMultipv = int(Options["MultiPV"]);
    setcmd("MultiPV 200");

    for (std::size_t index = start_pos; index < moves.size(); index += 2) {
        // 每次均从初始 FEN 重建局面，再回放实走之前的所有 ply，得到本次待评估的局面。
        states = StateListPtr(new std::deque<StateInfo>(1));
        pos.set(variants.find(Options["UCI_Variant"])->second, fen, Options["UCI_Chess960"],
                &states->back(), Threads.main(), false);
        std::size_t i = 0;
        for (; i < index; ++i) {
            states->emplace_back();
            pos.do_move(UCI::to_move(pos, moves[i]), states->back());
        }
        // 防御性检查：确保 index 对应的待评估实走仍存在。
        if (i >= moves.size())
            break;
        // sync_cout << pos << sync_endl;
        std::istringstream gois("go depth 10");

        // 1) 用强裁判在当前局面做 MultiPV 搜索。
        // rootMoves 会按裁判认为的着法质量从好到坏排序，因此可以直接得到：
        //   - best_move / best_cp / best_win_rate
        //   - oracle_candidates 全候选列表
        //   - played_rank / topN_hit / num_better_moves
        // 注意：MultiPV 上限设为 200，象棋常见局面通常足够覆盖全部合法着；
        // 如果未来遇到候选被截断，可以通过 candidate_count 和合法着数对比发现。
        go(pos, gois, states);
        Threads.main()->wait_for_search_finished();
        auto bestThread = Threads.get_best_thread();
        const Search::RootMoves rootMoves = bestThread->rootMoves;
        if (rootMoves.empty() || rootMoves[0].pv.empty())
            continue;

        const std::string best_move = UCI::move(pos, rootMoves[0].pv[0]);
        const Value bestValue = rootMoves[0].score;
        std::tuple<int, int, int> bestWv;
        UCI::wdl(bestValue, pos.game_ply(), bestWv);
        const int bestWin = std::get<0>(bestWv);
        const double bestCp = UCI::value_cp(bestValue);

        MoveRecord2 rec;
        // human_index 从 1 开始计数；ply_index 保留整盘走法序列中的 1 起始位置。
        rec.human_index = int((index - start_pos) / 2 + 1);
        rec.ply_index = int(index + 1);
        rec.fen_before = pos.fen();
        rec.move = moves[index];
        rec.best_move = best_move;
        rec.best_cp = bestCp;
        rec.best_win_rate = bestWin / 10.0;
        fill_oracle_candidate_features(rec, pos, rootMoves);

        // 在 MultiPV 结果中定位实走。找不到时保留对应 optional 字段为空，
        // 表示候选列表可能被 MultiPV 上限截断或该走法无法转换。
        double playedCp = 0.0;
        int playedWin = 0;
        bool playedFound = false;

        // 2) MultiPV 200 覆盖当前局面合法着，实走应当在候选中。
        // 直接从候选取实走分数，避免再次 searchmoves 引入额外耗时和缓存扰动。
        for (const auto &rm : rootMoves) {
            if (rm.pv.empty())
                continue;
            if (UCI::move(pos, rm.pv[0]) != moves[index])
                continue;

            std::tuple<int, int, int> playedWv;
            UCI::wdl(rm.score, pos.game_ply(), playedWv);
            playedWin = std::get<0>(playedWv);
            playedCp = UCI::value_cp(rm.score);
            playedFound = true;
            break;
        }

        if (playedFound) {
            rec.played_cp = playedCp;
            rec.played_win_rate = playedWin / 10.0;
            rec.loss = std::max(0.0, (bestWin - playedWin) / 10.0);
        }

        records.push_back(rec);
    }

    // 不将临时搜索配置泄漏给后续 UCI 命令。
    setcmd("MultiPV " + std::to_string(oldMultipv));

    // 以单条协议输出返回全部回合的分析记录，供调用端解析。
    nlohmann::json recs = nlohmann::json::array();
    for (const auto &r : records)
        recs.push_back(move_record_to_json(r));
    nlohmann::json j;
    j["records"] = std::move(recs);
    sync_cout << "estimate2result " << j.dump() << sync_endl;
}

void multipv(Position &pos, istringstream &is, StateListPtr &states,
             const std::vector<Move> &banmoves = {})
{
    auto setcmd = [](const std::string &cmd) {
        istringstream iss(cmd);
        setoption(iss);
    };
    static constexpr const char *set_multicout_on = "MultiMoves true";
    static constexpr const char *set_multicout_off = "MultiMoves ";
    static constexpr const char *set_multipv_on = "MultiPV 200";
    static constexpr const char *set_multipv_off = "MultiPV ";
    int oldMultipv = int(Options["MultiPV"]);
    std::string oldMulticout = std::string(Options["MultiMoves"]);
    setcmd(set_multicout_on);
    setcmd(set_multipv_on);
    go(pos, is, states, banmoves);
    Threads.main()->wait_for_search_finished();
    setcmd(set_multicout_off + oldMulticout);
    setcmd(std::string(set_multipv_off) + std::to_string(oldMultipv));
}

// bench() is called when engine receives the "bench" command. Firstly
// a list of UCI commands is setup according to bench parameters, then
// it is run one by one printing a summary at the end.

void bench(Position &pos, istream &args, StateListPtr &states)
{

    string token;
    uint64_t num, nodes = 0, cnt = 1;

    vector<string> list = setup_bench(pos, args);
    num = count_if(list.begin(), list.end(),
                   [](string s) { return s.find("go ") == 0 || s.find("eval") == 0; });

    TimePoint elapsed = now();

    for (const auto &cmd : list) {
        istringstream is(cmd);
        is >> skipws >> token;

        if (token == "go" || token == "eval") {
            cerr << "\nPosition: " << cnt++ << '/' << num << " (" << pos.fen() << ")" << endl;
            if (token == "go") {
                go(pos, is, states);
                Threads.main()->wait_for_search_finished();
                nodes += Threads.nodes_searched();
            } else
                trace_eval(pos);
        } else if (token == "setoption")
            setoption(is);
        else if (token == "position")
            position(pos, is, states);
        else if (token == "ucinewgame") {
            Search::clear();
            elapsed = now();
        }  // Search::clear() may take some while
    }

    elapsed = now() - elapsed + 1;  // Ensure positivity to avoid a 'divide by zero'

    dbg_print();  // Just before exiting

    cerr << "\n==========================="
         << "\nTotal time (ms) : " << elapsed << "\nNodes searched  : " << nodes
         << "\nNodes/second    : " << 1000 * nodes / elapsed << endl;
}

// The win rate model returns the probability (per mille) of winning given an eval
// and a game-ply. The model fits rather accurately the LTC fishtest statistics.
int win_rate_model(Value v, int ply)
{

    // The model captures only up to 240 plies, so limit input (and rescale)
    double m = std::min(240, ply) / 64.0;

    // Coefficients of a 3rd order polynomial fit based on fishtest data
    // for two parameters needed to transform eval to the argument of a
    // logistic function.
    double as[] = { -3.68389304, 30.07065921, -60.52878723, 149.53378557 };
    double bs[] = { -2.0181857, 15.85685038, -29.83452023, 47.59078827 };
    double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
    double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

    // Transform eval to centipawns with limited range
    double x = std::clamp(double(100 * v) / PawnValueEg, -2000.0, 2000.0);

    // Return win rate in per mille (rounded to nearest)
    return int(0.5 + 1000 / (1 + std::exp((a - x) / b)));
}

// load() is called when engine receives the "load" command.
// The function reads variant configuration files.

void load(istringstream &is)
{

    string token;
    std::getline(is >> std::ws, token);
    std::size_t end = token.find_last_not_of(' ');
    if (end != std::string::npos)
        Options["VariantPath"] = token.erase(end + 1);
}

// check() is called when engine receives the "check" command.
// The function reads a variant configuration file and validates it.

void check(istringstream &is)
{

    string token;
    std::getline(is >> std::ws, token);
    std::size_t end = token.find_last_not_of(' ');
    if (end != std::string::npos)
        variants.parse<true>(token.erase(end + 1));
}

}  // namespace

/// UCI::loop() waits for a command from stdin, parses it and calls the appropriate
/// function. Also intercepts EOF from stdin to ensure gracefully exiting if the
/// GUI dies unexpectedly. When called with some command line arguments, e.g. to
/// run 'bench', once the command is executed the function returns immediately.
/// In addition to the UCI ones, also some additional debug commands are supported.

void UCI::loop(int argc, char *argv[])
{

    Position pos;
    string token, cmd;
    StateListPtr states(new std::deque<StateInfo>(1));

    assert(variants.find(Options["UCI_Variant"])->second != nullptr);
    pos.set(variants.find(Options["UCI_Variant"])->second,
            variants.find(Options["UCI_Variant"])->second->startFen, false, &states->back(),
            Threads.main());

    for (int i = 1; i < argc; ++i)
        cmd += std::string(argv[i]) + " ";

    // XBoard state machine
    XBoard::stateMachine = new XBoard::StateMachine(pos, states);
    // UCCI banmoves state
    std::vector<Move> banmoves = {};

    if (argc > 1 && (std::strcmp(argv[1], "noautoload") == 0)) {
        cmd = "";
        argc = 1;
    } else if (argc == 1 || !(std::strcmp(argv[1], "load") == 0)) {
        // Check environment for variants.ini file
        char *envVariantPath = std::getenv("FAIRY_STOCKFISH_VARIANT_PATH");
        if (envVariantPath != NULL)
            Options["VariantPath"] = std::string(envVariantPath);
    }

    do {
        if (argc == 1 && !getline(cin, cmd))  // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear();  // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        if (token == "quit" || token == "stop")
            Threads.stop = true;

        // The GUI sends 'ponderhit' to tell us the user has played the expected move.
        // So 'ponderhit' will be sent if we were told to ponder on the same move the
        // user has played. We should continue searching but switch from pondering to
        // normal search.
        else if (token == "ponderhit")
            Threads.main()->ponder = false;  // Switch to normal search

        else if (token == "uci" || token == "usi" || token == "ucci" || token == "xboard") {
            Options["Protocol"].set_default(token);
            string defaultVariant = string(
#ifdef LARGEBOARDS
                token == "usi"    ? "shogi"
                : token == "ucci" ? "xiangqi"
#else
                token == "usi"    ? "minishogi"
                : token == "ucci" ? "minixiangqi"
#endif
                                  : "chess");
            Options["UCI_Variant"].set_default(defaultVariant);
            std::istringstream ss("startpos");
            position(pos, ss, states);
            if (token == "uci" || token == "usi" || token == "ucci")
                sync_cout << "id name " << engine_info(true) << "\n"
                          << Options << "\n"
                          << token << "ok" << sync_endl;
        }

        else if (Options["Protocol"] == "xboard")
            XBoard::stateMachine->process_command(token, is);

        else if (token == "setoption")
            setoption(is);
        // UCCI-specific banmoves command
        else if (token == "banmoves")
            while (is >> token)
                banmoves.push_back(UCI::to_move(pos, token));
        else if (token == "go")
            go(pos, is, states, banmoves);
        else if (token == "multipv")
            multipv(pos, is, states, banmoves);
        else if (token == "estimate") {
            bool humanRed = true;
            std::streampos posBeforeColor = is.tellg();
            std::string colorToken;
            if (is >> colorToken) {
                if (colorToken == "red")
                    humanRed = true;
                else if (colorToken == "black")
                    humanRed = false;
                else
                    is.seekg(posBeforeColor);
            }
            estimate(is, humanRed);
        } else if (token == "estimate2") {
            bool humanRed = true;
            std::streampos posBeforeColor = is.tellg();
            std::string colorToken;
            if (is >> colorToken) {
                if (colorToken == "red")
                    humanRed = true;
                else if (colorToken == "black")
                    humanRed = false;
                else
                    is.seekg(posBeforeColor);
            }
            estimate2(is, humanRed);
        } else if (token == "position")
            position(pos, is, states), banmoves.clear();
        else if (token == "ucinewgame" || token == "usinewgame" || token == "uccinewgame")
            Search::clear();
        else if (token == "isready")
            sync_cout << "readyok" << sync_endl;

        // Additional custom non-UCI commands, mainly for debugging.
        // Do not use these commands during a search!
        else if (token == "flip")
            pos.flip();
        else if (token == "bench")
            bench(pos, is, states);
        else if (token == "d")
            sync_cout << pos << sync_endl;
        else if (token == "eval")
            trace_eval(pos);
        else if (token == "compiler")
            sync_cout << compiler_info() << sync_endl;
        else if (token == "export_net") {
            std::optional<std::string> filename;
            std::string f;
            if (is >> skipws >> f)
                filename = f;
            Eval::NNUE::save_eval(filename);
        } else if (token == "load") {
            load(is);
            argc = 1;
        }  // continue reading stdin
        else if (token == "check")
            check(is);
        // UCI-Cyclone omits the "position" keyword
        else if (token == "fen" || token == "startpos") {
#ifdef LARGEBOARDS
            if (Options["Protocol"] == "uci" && Options["UCI_Variant"] == "chess") {
                Options["Protocol"].set_default("ucicyclone");
                Options["UCI_Variant"].set_default("xiangqi");
            }
#endif
            is.seekg(0);
            position(pos, is, states);
        } else if (!token.empty() && token[0] != '#')
            sync_cout << "Unknown command: " << cmd << sync_endl;

    } while (token != "quit" && argc == 1);  // Command line args are one-shot
}

/// UCI::value() converts a Value to a string suitable for use with the UCI
/// protocol specification:
///
/// cp <x>    The score from the engine's point of view in centipawns.
/// mate <y>  Mate in y moves, not plies. If the engine is getting mated
///           use negative values for y.

string UCI::value(Value v)
{

    assert(-VALUE_INFINITE < v && v < VALUE_INFINITE);

    stringstream ss;

    if (Options["Protocol"] == "xboard") {
        if (abs(v) < VALUE_MATE_IN_MAX_PLY)
            ss << v * 100 / PawnValueEg;
        else
            ss << (v > 0 ? XBOARD_VALUE_MATE + VALUE_MATE - v + 1
                         : -XBOARD_VALUE_MATE - VALUE_MATE - v - 1) /
                      2;
    } else

        if (abs(v) < VALUE_MATE_IN_MAX_PLY)
        ss << "cp " << v * 100 / PawnValueEg;
    else if (Options["Protocol"] == "usi")
        // In USI, mate distance is given in ply
        ss << "mate " << (v > 0 ? VALUE_MATE - v : -VALUE_MATE - v);
    else
        ss << "mate " << (v > 0 ? VALUE_MATE - v + 1 : -VALUE_MATE - v - 1) / 2;

    return ss.str();
}

double UCI::value_cp(Value v)
{
    assert(-VALUE_INFINITE < v && v < VALUE_INFINITE);
    double cp(0);
    if (Options["Protocol"] == "xboard") {
        if (abs(v) < VALUE_MATE_IN_MAX_PLY)
            cp = v * 100 / PawnValueEg;
        else
            cp = (v > 0 ? XBOARD_VALUE_MATE + VALUE_MATE - v + 1
                        : -XBOARD_VALUE_MATE - VALUE_MATE - v - 1) /
                 2;
    } else if (abs(v) < VALUE_MATE_IN_MAX_PLY)
        cp = v * 100 / PawnValueEg;
    else if (Options["Protocol"] == "usi")
        // In USI, mate distance is given in ply
        cp = (v > 0 ? VALUE_MATE - v : -VALUE_MATE - v);
    else
        cp = (v > 0 ? VALUE_MATE - v + 1 : -VALUE_MATE - v - 1) / 2;

    return cp;
}

/// UCI::wdl() report WDL statistics given an evaluation and a game ply, based on
/// data gathered for fishtest LTC games.

string UCI::wdl(Value v, int ply)
{

    stringstream ss;

    int wdl_w = win_rate_model(v, ply);
    int wdl_l = win_rate_model(-v, ply);
    int wdl_d = 1000 - wdl_w - wdl_l;
    ss << " wdl " << wdl_w << " " << wdl_d << " " << wdl_l;

    return ss.str();
}

void UCI::wdl(Value v, int ply, std::tuple<int, int, int> &wv)
{

    stringstream ss;

    int wdl_w = win_rate_model(v, ply);
    int wdl_l = win_rate_model(-v, ply);
    int wdl_d = 1000 - wdl_w - wdl_l;
    std::get<0>(wv) = wdl_w;
    std::get<1>(wv) = wdl_d;
    std::get<2>(wv) = wdl_l;
}

/// UCI::square() converts a Square to a string in algebraic notation (g1, a7, etc.)

std::string UCI::square(const Position &pos, Square s)
{
#ifdef LARGEBOARDS
    if (Options["Protocol"] == "usi")
        return rank_of(s) < RANK_10
                   ? std::string{ char('1' + pos.max_file() - file_of(s)),
                                  char('a' + pos.max_rank() - rank_of(s)) }
                   : std::string{ char('0' + (pos.max_file() - file_of(s) + 1) / 10),
                                  char('0' + (pos.max_file() - file_of(s) + 1) % 10),
                                  char('a' + pos.max_rank() - rank_of(s)) };
    else if (pos.max_rank() == RANK_10 && Options["Protocol"] != "uci")
        return std::string{ char('a' + file_of(s)), char('0' + rank_of(s)) };
    else
        return rank_of(s) < RANK_10
                   ? std::string{ char('a' + file_of(s)), char('1' + (rank_of(s) % 10)) }
                   : std::string{ char('a' + file_of(s)), char('0' + ((rank_of(s) + 1) / 10)),
                                  char('0' + ((rank_of(s) + 1) % 10)) };
#else
    return Options["Protocol"] == "usi"
               ? std::string{ char('1' + pos.max_file() - file_of(s)),
                              char('a' + pos.max_rank() - rank_of(s)) }
               : std::string{ char('a' + file_of(s)), char('1' + rank_of(s)) };
#endif
}

/// UCI::dropped_piece() generates a piece label string from a Move.

string UCI::dropped_piece(const Position &pos, Move m)
{
    assert(type_of(m) == DROP);
    if (dropped_piece_type(m) == pos.promoted_piece_type(in_hand_piece_type(m)))
        // Dropping as promoted piece
        return std::string{ '+', pos.piece_to_char()[in_hand_piece_type(m)] };
    else
        return std::string{ pos.piece_to_char()[dropped_piece_type(m)] };
}

/// UCI::move() converts a Move to a string in coordinate notation (g1f3, a7a8q).
/// The only special case is castling, where we print in the e1g1 notation in
/// normal chess mode, and in e1h1 notation in chess960 mode. Internally all
/// castling moves are always encoded as 'king captures rook'.

string UCI::move(const Position &pos, Move m)
{

    Square from = from_sq(m);
    Square to = to_sq(m);

    if (m == MOVE_NONE)
        return Options["Protocol"] == "usi" ? "resign" : "(none)";

    if (m == MOVE_NULL)
        return "0000";

    if (is_pass(m) && Options["Protocol"] == "xboard")
        return "@@@@";

    if (is_gating(m) && gating_square(m) == to)
        from = to_sq(m), to = from_sq(m);
    else if (type_of(m) == CASTLING && !pos.is_chess960()) {
        to = make_square(to > from ? pos.castling_kingside_file() : pos.castling_queenside_file(),
                         rank_of(from));
        // If the castling move is ambiguous with a normal king move, switch to 960 notation
        if (pos.pseudo_legal(make_move(from, to)))
            to = to_sq(m);
    }

    string move = (type_of(m) == DROP
                       ? UCI::dropped_piece(pos, m) + (Options["Protocol"] == "usi" ? '*' : '@')
                       : UCI::square(pos, from)) +
                  UCI::square(pos, to);

    if (type_of(m) == PROMOTION)
        move += pos.piece_to_char()[make_piece(BLACK, promotion_type(m))];
    else if (type_of(m) == PIECE_PROMOTION)
        move += '+';
    else if (type_of(m) == PIECE_DEMOTION)
        move += '-';
    else if (is_gating(m)) {
        move += pos.piece_to_char()[make_piece(BLACK, gating_type(m))];
        if (gating_square(m) != from)
            move += UCI::square(pos, gating_square(m));
    }

    return move;
}

/// UCI::to_move() converts a string representing a move in coordinate notation
/// (g1f3, a7a8q) to the corresponding legal Move, if any.

Move UCI::to_move(const Position &pos, string &str)
{

    if (str.length() == 5) {
        if (str[4] == '=')
            // shogi moves refraining from promotion might use equals sign
            str.pop_back();
        else
            // Junior could send promotion piece in uppercase
            str[4] = char(tolower(str[4]));
    }

    for (const auto &m : MoveList<LEGAL>(pos))
        if (str == UCI::move(pos, m) ||
            (is_pass(m) && str == UCI::square(pos, from_sq(m)) + UCI::square(pos, to_sq(m))))
            return m;

    return MOVE_NONE;
}

}  // namespace Stockfish
