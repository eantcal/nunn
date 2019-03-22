//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_QLEARN_H__
#define __NU_QLEARN_H__


/* -------------------------------------------------------------------------- */

#include "nu_learner_listener.h"

#include <unordered_map>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<
    class A, 
    class S, 
    class AG, 
    class P, 
    class T = double,
    class ARM = std::unordered_map<A, T>,
    class Q = std::unordered_map<S, ARM>>
class qlearn_t {
public:
    using real_t = T;
    using action_t = A;       // possible action in state S
    using state_t = S;        // representation of the environmental states
    using agent_t = AG;       // the Q-learning agent
    using policy_t = P;       // policy to follow
    using reward_t = real_t;  // the reward
    using listener_t = learner_listener_t<real_t>;

    using action_reward_t = ARM;
    using q_map_t = Q;

    qlearn_t(listener_t * listener = nullptr) noexcept :
        _listener(listener)
    {}

    qlearn_t(const qlearn_t&) = default;
    qlearn_t& operator=(const qlearn_t&) = default;

    real_t get_learning_rate() const noexcept {
        return _learning_rate;
    }

    real_t get_discount_rate() const noexcept {
        return _discount_rate;
    }

    void set_learning_rate(const real_t& lr) const noexcept {
        _learning_rate = lr;
    }

    void set_discount_rate(const real_t& dr) const noexcept {
        _discount_rate = dr;
    }

    action_t select_action(
        const agent_t& agent, 
        const policy_t & policy = policy_t()) 
    {
        return policy.template get_learned_action<q_map_t>(agent, get_q());
    }

    // learn episode
    real_t learn(agent_t& agent, const policy_t & policy=policy_t()) {

        size_t move_cnt = 0;
        real_t reward = 0;

        while (!agent.goal()) {
            if (_listener && !_listener->notify(reward, move_cnt++)) {
                break;
            }

            reward += update_q(agent, policy);
        }

        return reward;
    }

    const q_map_t & get_q() const noexcept {
        return _q_map;
    }
    
protected:
    q_map_t & get_q() noexcept {
        return _q_map;
    }  

    real_t update_q(agent_t& agent, const policy_t & policy) {

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

        return qsa;
    }

private:
    real_t _learning_rate = 0.1;
    real_t _discount_rate = 0.9;

    q_map_t _q_map;
    policy_t _policy;

    listener_t * _listener = nullptr;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_QLEARN_H__

