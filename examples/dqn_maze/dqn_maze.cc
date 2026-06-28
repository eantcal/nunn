//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// dqn_maze — DQN demonstration on a small grid maze.
//
// Layout (5x5):
//   #####
//   #S..#    S = start (1,1)
//   #.#.#    # = wall
//   #..G#    G = goal (3,3)
//   #####
//
// State:   normalized grid position [col/(W-1), row/(H-1)]
// Actions: 0=Up, 1=Down, 2=Left, 3=Right
// Network: [2, 32, 32, 4] with Tanh hidden / Linear output
//
// Usage: dqn_maze [episodes=600] [lr=0.005] [seed=0]
//

#include "nu_dqn.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ── Maze ─────────────────────────────────────────────────────────────────────

constexpr int W = 5;
constexpr int H = 5;

// clang-format off
const char MAZE[H][W + 1] = {
    "#####",
    "#S..#",
    "#.#.#",
    "#..G#",
    "#####"
};
// clang-format on

constexpr int START_X = 1, START_Y = 1;
constexpr int GOAL_X = 3, GOAL_Y = 3;

bool isWall(int x, int y)
{
    if (x < 0 || x >= W || y < 0 || y >= H)
        return true;
    return MAZE[y][x] == '#';
}

// ── Environment ───────────────────────────────────────────────────────────────

constexpr int N_ACTIONS = 4; // 0=Up, 1=Down, 2=Left, 3=Right
const int DX[N_ACTIONS] = { 0, 0, -1, 1 };
const int DY[N_ACTIONS] = { -1, 1, 0, 0 };

struct StepResult {
    std::vector<double> s_next;
    double reward;
    bool done;
};

std::vector<double> encode(int x, int y)
{
    return { static_cast<double>(x) / (W - 1), static_cast<double>(y) / (H - 1) };
}

StepResult step(int x, int y, int action)
{
    const int nx = x + DX[action];
    const int ny = y + DY[action];

    if (isWall(nx, ny)) {
        // Stay in place, small penalty.
        return { encode(x, y), -0.1, false };
    }
    if (nx == GOAL_X && ny == GOAL_Y) {
        return { encode(nx, ny), 1.0, true };
    }
    return { encode(nx, ny), -0.01, false };
}

// ── Rendering ─────────────────────────────────────────────────────────────────

void printMaze(int ax, int ay)
{
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            if (col == ax && row == ay)
                std::cout << 'A';
            else if (col == GOAL_X && row == GOAL_Y)
                std::cout << 'G';
            else
                std::cout << MAZE[row][col];
        }
        std::cout << '\n';
    }
}

} // namespace

int main(int argc, char* argv[])
{
    const int EPISODES = argc > 1 ? std::stoi(argv[1]) : 600;
    const double LR = argc > 2 ? std::stod(argv[2]) : 0.005;

    std::cout << "DQN maze demo\n";
    std::cout << "  episodes=" << EPISODES << "  lr=" << LR << "\n\n";
    std::cout << "Map:\n";
    printMaze(START_X, START_Y);
    std::cout << '\n';

    using LC = nu::MlpMatrixNN::LayerConfig;
    const std::vector<nu::MlpMatrixNN::LayerConfig> layers{ LC(2), LC(32, nu::Activation::Tanh),
        LC(32, nu::Activation::Tanh), LC(static_cast<size_t>(N_ACTIONS), nu::Activation::Linear) };

    nu::Dqn dqn(layers, LR,
        /*bufferCapacity=*/5000,
        /*batchSize=*/32,
        /*gamma=*/0.99,
        /*targetUpdateFreq=*/100);

    constexpr int MAX_STEPS = 80;
    constexpr int REPORT = 50;

    int totalSuccesses = 0;
    int windowSuccesses = 0;

    std::cout << std::setw(8) << "Episode" << std::setw(10) << "Steps" << std::setw(10) << "Return"
              << std::setw(10) << "Success" << std::setw(14) << "LearnSteps"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (int ep = 1; ep <= EPISODES; ++ep) {
        const double eps = std::max(0.05, 1.0 - static_cast<double>(ep) / (0.7 * EPISODES));

        int x = START_X, y = START_Y;
        std::vector<double> s = encode(x, y);
        double ret = 0.0;
        bool success = false;

        for (int t = 0; t < MAX_STEPS; ++t) {
            const int a = dqn.selectAction(s, eps);
            auto [s_next, r, done] = step(x, y, a);

            dqn.learn(s, a, r, s_next, done);
            ret += r;

            if (done) {
                success = true;
                break;
            }

            // Update position.
            const int nx = x + DX[a];
            const int ny = y + DY[a];
            if (!isWall(nx, ny)) {
                x = nx;
                y = ny;
            }
            s = s_next;
        }

        if (success) {
            ++totalSuccesses;
            ++windowSuccesses;
        }

        if (ep % REPORT == 0) {
            std::cout << std::setw(8) << ep << std::setw(10) << (success ? "yes" : "-")
                      << std::fixed << std::setprecision(3) << std::setw(10) << ret << std::setw(9)
                      << windowSuccesses << "/" << REPORT << std::setw(10)
                      << dqn.getLearnStepCount() << "\n";
            windowSuccesses = 0;
        }
    }

    std::cout << "\nTotal successes: " << totalSuccesses << "/" << EPISODES << "\n";

    // Greedy trace from start.
    std::cout << "\nGreedy trace (epsilon=0):\n";
    int x = START_X, y = START_Y;
    for (int t = 0; t < MAX_STEPS; ++t) {
        printMaze(x, y);
        const int a = dqn.selectAction(encode(x, y), 0.0);
        const char* aname[] = { "Up", "Down", "Left", "Right" };
        std::cout << "  action: " << aname[a] << "\n\n";

        const int nx = x + DX[a];
        const int ny = y + DY[a];
        if (nx == GOAL_X && ny == GOAL_Y) {
            std::cout << "  Reached goal!\n";
            break;
        }
        if (!isWall(nx, ny)) {
            x = nx;
            y = ny;
        } else {
            std::cout << "  (wall — stopping trace)\n";
            break;
        }
    }

    return 0;
}
