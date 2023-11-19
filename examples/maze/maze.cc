//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/*
   Maze example.
*/

/* -------------------------------------------------------------------------- */

#include "nu_qlearn.h"
#include "nu_sarsa.h"
#include "nu_e_greedy_policy.h"
#include "nu_softmax_policy.h"


/* -------------------------------------------------------------------------- */

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#endif


/* -------------------------------------------------------------------------- */

#include <list>
#include <iostream>
#include <vector>
#include <thread>
#include <iomanip>


/* -------------------------------------------------------------------------- */

struct Envirnoment
{
    enum {_X=46, _Y=31};
    
    const bool map[_Y][_X] = {
    { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 },
    { 1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1 },
    { 1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1 },
    { 1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1 },
    { 1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1 },
    { 1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1 },
    { 1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1 },
    { 1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1 },
    { 1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1 },
    { 1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1 },
    { 1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1 },
    { 1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1 },
    { 1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1 },
    { 1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1 },
    { 1,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1 },
    { 1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1 },
    { 1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1 },
    { 1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1 },
    { 1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1 },
    { 1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1 },
    { 1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1 },
    { 1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1 },
    { 1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1 },
    { 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1 },
    { 1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,1 },
    { 1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1 },
    { 1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1 },
    { 1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1 },
    { 1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1 },
    { 1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1 },
    { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 } };
    
    Envirnoment() = default;

    constexpr int max_x() const { return _X; }
    constexpr int max_y() const { return _Y; }
};


/* -------------------------------------------------------------------------- */

struct Action
{
    enum class move_t : int { Left, Right, Up, Down };

    using list_t = std::vector<Action>;

    Action(const move_t& m) noexcept : _move(m) {}

    Action(const Action&) = default;
    Action& operator=(const Action&) = default;

    const move_t& get() const noexcept {
        return _move;
    }

    bool operator==(const Action& other) const noexcept {
        return _move == other._move;
    }
    
    static list_t make_complete_list() noexcept {
        return {
            Action(move_t::Left),
            Action(move_t::Right),
            Action(move_t::Up),
            Action(move_t::Down)
        };
    }

private:
    move_t _move;
};


/* -------------------------------------------------------------------------- */

struct State
{
public:
    State() = default;

    State(const int& x, const int& y) noexcept : _x(x), _y(y) {}

    State(const State&) = default;
    State& operator=(const State&) = default;

    int get_x() const noexcept { return _x; }
    int get_y() const noexcept { return _y; }

    void apply(const Action& action) noexcept {
        if (action.get() == Action::move_t::Left) {
            --_x;
        }
        else if (action.get() == Action::move_t::Right) {
            ++_x;
        }
        else if (action.get() == Action::move_t::Up) {
            --_y;
        }
        else if (action.get() == Action::move_t::Down) {
            ++_y;
        }
    }

    bool operator==(const State& other) const noexcept {
        return _x == other._x && _y == other._y;
    }

private:
    int _x = 0;
    int _y = 0;
};


/* -------------------------------------------------------------------------- */

class Agent {
public:
    Agent(
        const Envirnoment & env, 
        const State& init, 
        const State& goal) noexcept 
        : 
        _env(env),
        _state(init),
        _goalState(goal)
    {}

    bool isValid(const Action& action) const noexcept {
        const auto x = _state.get_x();
        const auto y = _state.get_y();

        if (action.get() == Action::move_t::Left) {
            return (x > 0 && _env.map[y][x - 1] == false);
        }
        else if (action.get() == Action::move_t::Right) {
            return (x < (_env.max_x()-1) && _env.map[y][x + 1] == false);
        }
        else if (action.get() == Action::move_t::Up) {
            return (y > 0 && _env.map[y - 1][x] == false);
        }
        else if (action.get() == Action::move_t::Down) {
            return (y < (_env.max_y()-1) && _env.map[y + 1][x] == false);
        }

        return false;
    }

    Action::list_t getValidActions() const noexcept {
        const auto list = Action::make_complete_list();

        Action::list_t vlist;

        for (const auto& action : list) {
            if (isValid(action)) {
                vlist.push_back(action);
            }
        }

        return vlist;
    }

    bool doAction(const Action& action) {

        if (isValid(action)) {
            _state.apply(action);
            return true;
        }

        return false;
    }

    const State& getCurrentState() const noexcept {
        return _state;
    }

    const State& getGoalState() const noexcept {
        return _goalState;
    }

    void setCurrentState(const State& state) noexcept {
        _state = state;
    }

    void setGoalState(const State& state) noexcept {
        _goalState = state;
    }

    const Envirnoment& getEnv() const noexcept {
        return _env;
    }

    bool goal() const noexcept {
        return _state == _goalState;
    }

    double reward() const noexcept {
        return goal() ? 100.0 : 0;
    }

private:
    const Envirnoment & _env;
    State _state;
    State _goalState;
};


/* -------------------------------------------------------------------------- */

struct Render
{
    void show(const Agent& a, std::ostream & os) const {
        const auto x = a.getCurrentState().get_x();
        const auto y = a.getCurrentState().get_y();

        const auto gx = a.getGoalState().get_x();
        const auto gy = a.getGoalState().get_y();

        const auto& env = a.getEnv();

        for (auto row = 0; row < env.max_y(); ++row) {
            for (auto col = 0; col < env.max_x(); ++col) {

                if (x == gx && y == gy && gx == col && gy == row) {
                    os << "$";
                }
                else if (x == col && y == row) {
                    os << "A";
                }
                else if (gx == col && gy == row) {
                    os << "G";
                }
                else {
                    if (env.map[row][col]) {
                        os << '*';
                    }
                    else {
                        os << " ";
                    }
                }
            }
            os << std::endl;
        }
    }
};


/* -------------------------------------------------------------------------- */

namespace std {

template <>
struct hash<State>
{
    std::size_t operator()(const State& k) const
    {
        return std::hash<int>{} (k.get_x()) ^ 
               std::hash<int>{} (k.get_y() << 1);
    }
};


/* -------------------------------------------------------------------------- */

template <>
struct hash<Action>
{
    std::size_t operator()(const Action& k) const
    {
        return std::hash<int>{}(int(k.get()));
    }
};

} // std


/* -------------------------------------------------------------------------- */

void locate(int y, int x)
{
#ifdef _WIN32
    COORD c = { short((x - 1) & 0xffff), short((y - 1) & 0xffff) };
    ::SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), c);
#else
    printf("%c[%d;%df", 0x1B, y, x);
#endif
}


/* -------------------------------------------------------------------------- */

static void cls()
{
#ifdef _WIN32
    ::system("cls"); 
#else
    auto res = ::system("clear"); 
    (void) res;
#endif
}


/* -------------------------------------------------------------------------- */
#define E_GREEDY_POLICY
#ifdef E_GREEDY_POLICY
using Policy = nu::EGreedyPolicy<Action, Agent>;
#else
using Policy = nu::SoftmaxPolicy<Action, Agent>;
#endif

#define USE_SARSA
// define USE_SARSA to switch over Sarsa algorithm at compile time
#ifdef USE_SARSA
using Learner = 
    nu::Sarsa<
        Action, 
        State, 
        Agent, 
        Policy>;

#else // USE_QLEARN
using Learner = 
    nu::QLearn<
        Action, 
        State, 
        Agent, 
        Policy>;
#endif


/* -------------------------------------------------------------------------- */

struct simulator_t {

    template<class Render>
    size_t play(
        int episode,
        const Render& r,
        const Envirnoment& env,
        const State & goal,
        Learner & ql,
        int timeout)
    {
        size_t moveCnt = 0;

        State st(1, 1);
        Agent agent(env, st, goal);

        while (!agent.goal() && timeout--) {

            locate(1, 1);
            std::cout 
                << "Episode #" 
                << episode 
                << "                                 ";

            std::cout << std::endl;
            r.show(agent, std::cout);

            auto vlist = agent.getValidActions();

            if (!vlist.empty()) {
                auto action = ql.selectAction(agent);

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
            std::cout
                << "Episode #" << episode
                << " not completed: timeout! " << std::endl;
        }
        else {
            locate(1, 1);
            std::cout
                << "Episode #" << episode
                << " completed in " << moveCnt
                << " moves" << std::endl;
        }

        r.show(agent, std::cout);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        return moveCnt;
    }

};


/* -------------------------------------------------------------------------- */

int main()
{
    // envirnoment
    Envirnoment env;
    State goal(44,29);
    Render r;
    Learner ql;

    const int episodies = 100000;
    const int timeout = 3000;
    const double greward = 1000;

    cls();

    simulator_t simulator;

    auto line=[](size_t n) {
        while (n-->0) {
            std::cout << "-";
        }
        std::cout << std::endl;
    };

    std::cout << "Learning... " << std::endl;

    int episode = 0;
    for (;  episode < episodies; ++episode) {
        State st(1, 1);
        Agent agent(env, st, goal);

        auto reward = size_t(ql.learn(agent));
                
        std::cout << std::setw(5) << reward << " ";
        line(size_t(10 * log(double(reward))));

        if (reward>greward) {
            break;
        }
    }

    while (true) {
        simulator.play<Render>(
            episode, r, env, goal, ql, timeout);
    }
    
    return 0;
}


