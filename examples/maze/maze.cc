//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

// Maze example.

#include "nu_e_greedy_policy.h"
#include "nu_qlearn.h"
#include "nu_sarsa.h"
#include "nu_softmax_policy.h"

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#endif

#include <iomanip>
#include <iostream>
#include <list>
#include <thread>
#include <vector>
#include <algorithm>

struct Envirnoment {
    enum {
        _X = 46,
        _Y = 31
    };

private:
    // clang-format off
    const char* map_=
        "**********************************************"
        "*           *     *     *        *     *     *"
        "*  **********  *  *  *  *  ****  ****  *  ****"
        "*  *           *  *  *  *  *  *     *  *     *"
        "*  *  **********  ****  *  *  ****  *  ****  *"
        "*        *     *     *  *  *  *     *        *"
        "*  *******  *  ****  *  *  *  *  *******  ****"
        "*  *        *        *  *  *     *        *  *"
        "*  *  ****************  *  *******  *******  *"
        "*  *  *        *        *  *     *  *        *"
        "****  *  ****  *  *******  *  *  *  *  ****  *"
        "*     *     *  *  *        *  *     *     *  *"
        "*  *******  *  *  *  *******  *************  *"
        "*  *     *  *     *        *              *  *"
        "*  *  *  *  *************  ****  *******  *  *"
        "*     *  *  *        *     *           *  *  *"
        "*******  *  *  ****  *  ****************  *  *"
        "*        *        *     *        *  *        *"
        "*  **********  *  *******  ****  *  *  *  *  *"
        "*  *           *  *     *     *        *  *  *"
        "*  *  *******  *  *  *  **********  ****  *  *"
        "*  *        *  *  *  *     *        *     *  *"
        "*  **********  *  *  ****  *  *******  *******"
        "*              *     *  *     *        *     *"
        "*  *******************  *******  *  ****  *  *"
        "*  *                 *     *     *        *  *"
        "*  ****  *******  *******  *  *************  *"
        "*  *     *           *     *     *        *  *"
        "*  *  **********  *  *  *******  *  *******  *"
        "*     *           *     *        *           *"
        "**********************************************";
    // clang-format on

public:    
    Envirnoment() = default;

    bool getMapVal(int y, int x) const {
        return map_[x+y*_X] != ' ';
    }

    char getMapChar(int y, int x) const {
        return map_[x+y*_X];
    }

    constexpr int max_x() const { return _X; }
    constexpr int max_y() const { return _Y; }
};

struct Action {
    // Enum representing possible movements
    enum class move_t : int {
        Left,
        Right,
        Up,
        Down
    };

    // Using std::array for fixed-size list of actions
    using list_t = std::vector<Action>;

    // Explicit constructor to prevent implicit conversions
    explicit Action(move_t m) noexcept
        : _move(m)
    {
    }

    // Defaulted copy constructor and assignment operator
    Action() = default;
    Action(const Action&) = default;
    Action& operator=(const Action&) = default;

    // Getter method for move
    move_t get() const noexcept { return _move; }

    // Equality comparison operator
    bool operator==(const Action& other) const noexcept
    {
        return _move == other._move;
    }

    // Static method to generate a complete list of possible actions
    static list_t make_complete_list() noexcept
    {
        return { { Action(move_t::Left),
            Action(move_t::Right),
            Action(move_t::Up),
            Action(move_t::Down) } };
    }

private:
    move_t _move;
};

struct State {
public:
    State() = default;

    // Constructor using structured binding for clarity
    State(int x, int y) noexcept
        : _x(x)
        , _y(y)
    {
    }

    // Defaulted copy constructor and assignment operator
    State(const State&) = default;
    State& operator=(const State&) = default;

    // [[nodiscard]] attribute encourages checking the return value
    [[nodiscard]] int get_x() const noexcept { return _x; }
    [[nodiscard]] int get_y() const noexcept { return _y; }

    std::pair<int, int> getCoordinates() const noexcept { return { _x, _y }; }

    // Apply action to the state
    void apply(const Action& action) noexcept
    {
        switch (action.get()) {
        case Action::move_t::Left:
            --_x;
            break;
        case Action::move_t::Right:
            ++_x;
            break;
        case Action::move_t::Up:
            --_y;
            break;
        case Action::move_t::Down:
            ++_y;
            break;
        }
    }

    // Operator for comparing states
    bool operator==(const State& other) const noexcept
    {
        return _x == other._x && _y == other._y;
    }

private:
    int _x { 0 };
    int _y { 0 };
};

class Agent {
public:
    Agent(const Envirnoment& env, const State& init, const State& goal) noexcept
        : _env(env)
        , _state(init)
        , _goalState(goal)
    {
    }

    bool isValid(const Action& action) const noexcept
    {
        const auto x = _state.get_x();
        const auto y = _state.get_y();

        switch (action.get()) {
        case Action::move_t::Left:
            return (x > 0 && !_env.getMapVal(y, x - 1));
        case Action::move_t::Right:
            return (x < (_env.max_x() - 1) && !_env.getMapVal(y, x + 1));
        case Action::move_t::Up:
            return (y > 0 && !_env.getMapVal(y - 1, x));
        case Action::move_t::Down:
            return (y < (_env.max_y() - 1) && !_env.getMapVal(y + 1, x));
        default:
            return false;
        }
    }

    Action::list_t getValidActions() const noexcept
    {
        auto all_actions = Action::make_complete_list();
        Action::list_t valid_actions;
        std::copy_if(
            all_actions.cbegin(),
            all_actions.cend(),
            std::back_inserter(valid_actions),
            [this](const auto& action) { return isValid(action); });
        return valid_actions;
    }

    bool doAction(const Action& action)
    {
        if (isValid(action)) {
            _state.apply(action);
            return true;
        }
        return false;
    }

    [[nodiscard]] const State& getCurrentState() const noexcept
    {
        return _state;
    }

    [[nodiscard]] const State& getGoalState() const noexcept
    {
        return _goalState;
    }

    void setCurrentState(const State& state) noexcept { _state = state; }
    void setGoalState(const State& state) noexcept { _goalState = state; }

    [[nodiscard]] const Envirnoment& getEnv() const noexcept { return _env; }

    [[nodiscard]] bool goal() const noexcept { return _state == _goalState; }
    [[nodiscard]] double reward() const noexcept { return goal() ? 100.0 : 0; }

private:
    const Envirnoment& _env;
    State _state;
    State _goalState;
};

struct Render {
    static constexpr char AGENT_CHAR = 'A';
    static constexpr char GOAL_CHAR = 'G';
    static constexpr char AGENT_AT_GOAL_CHAR = '$';

    void show(const Agent& agent, std::ostream& os) const
    {
        const auto& [agentX, agentY] = agent.getCurrentState().getCoordinates();
        const auto& [goalX, goalY] = agent.getGoalState().getCoordinates();
        const auto& env = agent.getEnv();

        for (int row = 0; row < env.max_y(); ++row) {
            for (int col = 0; col < env.max_x(); ++col) {
                char displayChar = getDisplayChar(row, col, agentX, agentY, goalX, goalY, env);
                os << displayChar;
            }
            os << std::endl;
        }
    }

private:
    char getDisplayChar(int row,
        int col,
        int agentX,
        int agentY,
        int goalX,
        int goalY,
        const Envirnoment& env) const
    {
        if (row == agentY && col == agentX) {
            return (agentX == goalX && agentY == goalY) ? AGENT_AT_GOAL_CHAR
                                                        : AGENT_CHAR;
        } else if (row == goalY && col == goalX) {
            return GOAL_CHAR;
        } else {
            return env.getMapChar(row, col);
        }
    }
};

namespace std {

template <>
struct hash<State> {
    size_t operator()(const State& k) const
    {
        // A better hash combination technique using a prime number
        size_t h1 = std::hash<int> {}(k.get_x());
        size_t h2 = std::hash<int> {}(k.get_y());
        return h1 ^ (h2 << 1);
    }
};

template <>
struct hash<Action> {
    size_t operator()(const Action& k) const
    {
        // Hashing the underlying integer value of the enum
        return std::hash<size_t> {}(static_cast<size_t>(k.get()));
    }
};

} // namespace std


void locate(int y, int x)
{
#ifdef _WIN32
    COORD coord = { static_cast<SHORT>(x - 1), static_cast<SHORT>(y - 1) };
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
#else
    printf("\033[%d;%dH", y, x);
#endif
}

void cls()
{
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleScreenBufferInfo(hStdOut, &csbi)) {
        DWORD count;
        DWORD cellCount = csbi.dwSize.X * csbi.dwSize.Y;
        COORD homeCoords = { 0, 0 };
        FillConsoleOutputCharacter(hStdOut, ' ', cellCount, homeCoords, &count);
        FillConsoleOutputAttribute(
            hStdOut, csbi.wAttributes, cellCount, homeCoords, &count);
        SetConsoleCursorPosition(hStdOut, homeCoords);
    }
#else
    system("clear");
#endif
}

constexpr bool useEGreedyPolicy = true;
using Policy = std::conditional<useEGreedyPolicy,
    nu::EGreedyPolicy<Action, Agent>,
    nu::SoftmaxPolicy<Action, Agent>>::type;

constexpr bool useSarsa = true;
using Learner = std::conditional<useSarsa,
    nu::Sarsa<Action, State, Agent, Policy>,
    nu::QLearn<Action, State, Agent, Policy>>::type;

struct Simulator {
    template <class Render>
    size_t play(int episode,
        const Render& r,
        const Envirnoment& env,
        const State& goal,
        Learner& ql,
        int timeout)
    {
        size_t moveCnt = 0;

        State st(1, 1);
        Agent agent(env, st, goal);

        while (!agent.goal() && timeout--) {

            locate(1, 1);
            std::cout << "Episode #" << episode
                      << "                                 ";

            std::cout << std::endl;
            r.show(agent, std::cout);

            if (const auto vlist = agent.getValidActions(); !vlist.empty()) {
                const auto action = ql.selectAction(agent);

                if (!agent.doAction(action)) {
                    std::cerr << "Invalid operation, bug?!" << std::endl;
                    exit(1);
                }

                ++moveCnt;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (timeout < 1) {
            locate(1, 1);
            std::cout << "Episode #" << episode << " not completed: timeout! "
                      << std::endl;
        } else {
            locate(1, 1);
            std::cout << "Episode #" << episode << " completed in " << moveCnt
                      << " moves" << std::endl;
        }

        r.show(agent, std::cout);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        return moveCnt;
    }
};

struct App {
    // envirnoment
    Envirnoment env;
    State goal { 44, 29 };
    Render render;
    Learner learner;
    Simulator simulator;

    static constexpr int episodies { 100000 };
    static constexpr int timeout { 3000 };
    static constexpr double greward { 1000 };

    int learn()
    {
        const auto line = [](size_t n) {
            while (n-- > 0) {
                std::cout << "-";
            }
            std::cout << std::endl;
        };

        std::cout << "Learning... " << std::endl;

        for (int episode = 0; episode < episodies; ++episode) {
            State st(1, 1);
            Agent agent(env, st, goal);

            auto reward = size_t(learner.learn(agent));

            std::cout << std::setw(5) << reward << " ";
            line(size_t(10 * log(double(reward))));

            if (reward > greward) {
                return episode;
            }
        }

        return 0;
    }

    void play(int episode)
    {
        while (true) {
            simulator.play<Render>(
                episode, render, env, goal, learner, timeout);
        }
    }
};

int main()
{
    App app;
    const auto episode { app.learn() };
    app.play(episode);
    return 0;
}
