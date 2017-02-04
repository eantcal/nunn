/*
*  This file is part of nunnlib
*
*  nunnlib is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  nunnlib is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with nunnlib; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  US
*
*  Author: Antonino Calderone <acaldmail@gmail.com>
*
*/

/*
   Maze example.
*/

/* -------------------------------------------------------------------------- */

#include "nu_qlearn.h"
#include "nu_sarsa.h"
#include "nu_greedy_policy.h"


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


/* -------------------------------------------------------------------------- */

struct env_t
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
    
    constexpr int max_x() const { return _X; }
    constexpr int max_y() const { return _Y; }
};


/* -------------------------------------------------------------------------- */

struct action_t
{
    enum class move_t : int { Left, Right, Up, Down };

    using list_t = std::vector<action_t>;

    action_t(const move_t& m) noexcept : _move(m) {}

    action_t(const action_t&) = default;
    action_t& operator=(const action_t&) = default;

    const move_t& get() const noexcept {
        return _move;
    }

    bool operator==(const action_t& other) const noexcept {
        return _move == other._move;
    }
    
    static list_t make_complete_list() noexcept {
        return {
            action_t(move_t::Left),
            action_t(move_t::Right),
            action_t(move_t::Up),
            action_t(move_t::Down)
        };
    }

private:
    move_t _move;
};


/* -------------------------------------------------------------------------- */

struct state_t
{
public:
    state_t() = default;

    state_t(const int& x, const int& y) noexcept : _x(x), _y(y) {}

    state_t(const state_t&) = default;
    state_t& operator=(const state_t&) = default;

    int get_x() const noexcept { return _x; }
    int get_y() const noexcept { return _y; }

    void apply(const action_t& action) noexcept {
        if (action.get() == action_t::move_t::Left) {
            --_x;
        }
        else if (action.get() == action_t::move_t::Right) {
            ++_x;
        }
        else if (action.get() == action_t::move_t::Up) {
            --_y;
        }
        else if (action.get() == action_t::move_t::Down) {
            ++_y;
        }
    }

    bool operator==(const state_t& other) const noexcept {
        return _x == other._x && _y == other._y;
    }

private:
    int _x = 0;
    int _y = 0;
};


/* -------------------------------------------------------------------------- */

class agent_t
{
public:
    agent_t(
        const env_t & env, 
        const state_t& init, 
        const state_t& goal) noexcept 
        : 
        _env(env),
        _state(init),
        _goal_state(goal)
    {}

    bool is_valid(const action_t& action) const noexcept {
        const auto x = _state.get_x();
        const auto y = _state.get_y();

        if (action.get() == action_t::move_t::Left) {
            return (x > 0 && _env.map[y][x - 1] == false);
        }
        else if (action.get() == action_t::move_t::Right) {
            return (x < (_env.max_x()-1) && _env.map[y][x + 1] == false);
        }
        else if (action.get() == action_t::move_t::Up) {
            return (y > 0 && _env.map[y - 1][x] == false);
        }
        else if (action.get() == action_t::move_t::Down) {
            return (y < (_env.max_y()-1) && _env.map[y + 1][x] == false);
        }

        return false;
    }

    action_t::list_t valid_actions() const noexcept {
        const auto list = action_t::make_complete_list();

        action_t::list_t vlist;

        for (const auto& action : list) {
            if (is_valid(action)) {
                vlist.push_back(action);
            }
        }

        return vlist;
    }

    bool do_action(const action_t& action) {

        if (is_valid(action)) {
            _state.apply(action);
            return true;
        }

        return false;
    }

    const state_t& get_current_state() const noexcept {
        return _state;
    }

    const state_t& get_goal_state() const noexcept {
        return _goal_state;
    }

    void set_current_state(const state_t& state) noexcept {
        _state = state;
    }

    void set_goal_state(const state_t& state) noexcept {
        _goal_state = state;
    }

    const env_t& get_env() const noexcept {
        return _env;
    }

    bool goal() const noexcept {
        return _state == _goal_state;
    }

    double reward() const noexcept {
        return goal() ? 100.0 : 0;
    }

private:
    const env_t & _env;
    state_t _state;
    state_t _goal_state;
};


/* -------------------------------------------------------------------------- */

struct render_t
{
    void show(const agent_t& a, std::ostream & os) const
    {
        const auto x = a.get_current_state().get_x();
        const auto y = a.get_current_state().get_y();

        const auto gx = a.get_goal_state().get_x();
        const auto gy = a.get_goal_state().get_y();

        const auto& env = a.get_env();

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
struct hash<state_t>
{
    std::size_t operator()(const state_t& k) const
    {
        return std::hash<int>{} (k.get_x()) ^ 
               std::hash<int>{} (k.get_y() << 1);
    }
};


/* -------------------------------------------------------------------------- */

template <>
struct hash<action_t>
{
    std::size_t operator()(const action_t& k) const
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

// define USE_SARSA to switch over Sarsa algorithm at compile time

#ifdef USE_SARSA
using learner_t = 
    nu::sarsa_t<
        action_t, 
        state_t, 
        agent_t, 
        nu::greedy_policy_t<action_t, agent_t>>;

#else // USE_QLEARN
using learner_t = 
    nu::qlearn_t<
        action_t, 
        state_t, 
        agent_t, 
        nu::greedy_policy_t<action_t, agent_t>>;
#endif


/* -------------------------------------------------------------------------- */

struct simulator_t {

    template<class Render>
    size_t play(
        int episode,
        const Render& r,
        const env_t& env,
        const state_t & goal,
        learner_t & ql,
        int timeout)
    {
        size_t move_cnt = 0;

        state_t st(1, 1);
        agent_t agent(env, st, goal);

        while (!agent.goal() && timeout--) {

            locate(1, 1);
            std::cout 
                << "Episode #" 
                << episode 
                << "                                 ";

            std::cout << std::endl;
            r.show(agent, std::cout);

            auto vlist = agent.valid_actions();

            if (!vlist.empty()) {
                double reward = 0;
                auto action = ql.select_action(agent);

                if (!agent.do_action(action)) {
                    std::cerr << "Invalid operation, bug?!" << std::endl;
                    exit(1);
                }

                ++move_cnt;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(40));
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
                << " completed in " << move_cnt
                << " moves" << std::endl;
        }

        r.show(agent, std::cout);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        return move_cnt;
    }

};


/* -------------------------------------------------------------------------- */

int main()
{
    // envirnoment
    env_t env;
    state_t goal(44,29);
    render_t r;
    learner_t ql;

    const int episodies = 1000;
    const int timeout = 300;

    cls();

    simulator_t simulator;

    for (auto i = 0; i < episodies; ++i) {
        locate(1, 1);
        std::cout << "Learning... episode # " << i << "    " << std::endl;

        state_t st(1, 1);
        agent_t agent(env, st, goal);

        size_t moves = ql.learn(agent);
                
        locate(2, 1);
        std::cout 
            << "Solved in " << moves 
            << " moves                        " 
            << std::endl;

        if (i>170 && ((i % 15) == 0) ) {
           simulator.play<render_t>(
               i, r, env, goal, ql, timeout /* max moves before timeout !*/ );
        }
    }

    while (true) {
        simulator.play<render_t>(
            episodies, r, env, goal, ql, timeout);
    }
    
    return 0;
}


