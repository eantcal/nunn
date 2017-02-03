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


/* -------------------------------------------------------------------------- */

#ifndef __NU_QLEARN_H__
#define __NU_QLEARN_H__


/* -------------------------------------------------------------------------- */

#include <unordered_map>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<class A, class S, class AG, class P, class T=double>
class qlearn_t {
public:
    using value_t = double;
    using action_t = A;
    using state_t = S;
    using agent_t = AG;
    using policy_t = P;
    using reward_t = value_t;

    using action_reward_t = std::unordered_map<action_t, reward_t>;
    using q_map_t = std::unordered_map<state_t, action_reward_t>;

    virtual ~qlearn_t() {}

    qlearn_t() = default;
    qlearn_t(const qlearn_t&) = default;
    qlearn_t& operator=(const qlearn_t&) = default;

    value_t get_learning_rate() const noexcept {
        return _learning_rate;
    }

    value_t get_discount_rate() const noexcept {
        return _discount_rate;
    }

    void set_learning_rate(const value_t& lr) const noexcept {
        _learning_rate = lr;
    }

    void set_discount_rate(const value_t& dr) const noexcept {
        _discount_rate = dr;
    }

    virtual action_t select_action(
        const agent_t& agent, 
        const policy_t & policy = policy_t()) 
    {
        return policy.template get_learned_action<q_map_t>(agent, get_q());
    }

    // learn episode
    size_t learn(agent_t& agent, const policy_t & policy=policy_t()) {

        size_t move_cnt = 0;

        while (!agent.goal() && _continue(move_cnt)) {
            ++move_cnt;
            update_q(agent, policy);
        }

        return move_cnt;
    }

    const q_map_t & get_q() const noexcept {
        return _q_map;
    }
    
protected:
    q_map_t & get_q() noexcept {
        return _q_map;
    }  

    virtual void update_q(agent_t& agent, const policy_t & policy) {

        action_t action = 
            policy.template select_action<q_map_t>(agent, get_q());
        
        auto & qsa = get_q()[agent.get_current_state()][action];

        // update agent state
        agent.do_action(action); 

        // get current agent state
        const auto & agent_state = agent.get_current_state();

        // get a reward for current state
        const auto reward = agent.reward(); 

        // get a list of valid actions for current state
        const auto valid_actions = agent.valid_actions();

        auto max = get_q()[agent_state][valid_actions[0]];

        for (const auto & an_action : valid_actions) {
            auto val = get_q()[agent_state][an_action];
            if (val > max) {
                max = val;
            }
        }

        qsa += 
            get_learning_rate() * 
            (reward + get_discount_rate() * max - qsa);
    }

    // extending this class you can control learning loop
    // by redefinig the follow method
    virtual bool _continue(size_t /*move*/) {
        return true;
    }


private:
    value_t _learning_rate = 0.5;
    value_t _discount_rate = 0.5;

    q_map_t _q_map;

    policy_t _policy;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_QLEARN_H__

