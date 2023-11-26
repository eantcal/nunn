//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_qlgraph.h"

namespace nu
{
    assert(goal_state < n_of_states);

    QLGraph::QLGraph(
        const size_t &n_of_states,
        const size_t &goal_state,
        const Topology &topology)
        : _nOfStates(n_of_states),
          _goalState(goal_state),
          _rewardMtx(n_of_states),
          _q_mtx(n_of_states)
    {
        assert(goal_state < n_of_states);

        for (auto& destination : state.second) {
            _rewardMtx[state.first][destination] =
              destination == _goalState ? REWARD : NO_REWARD;
        }
    }
}

    bool QLGraph::learn(const size_t &nOfEpisodes, const Helper &helper)
    {
        for (size_t episode = 0; episode < nOfEpisodes; ++episode)
        {

        helper.beginEpisode(episode, *this);

        if (helper.quitRequestPending()) {
            return false;
        }

        auto _curState = size_t(helper.rnd() * double(_nOfStates)) % _nOfStates;

        bool goal = false;

        while (!goal) {

            if (helper.quitRequestPending()) {
                return false;
            }

            auto validActions = retrieveValidActions(_rewardMtx, _curState);
            const auto nOfActions = validActions.size();

            auto nextState =
              validActions[size_t(helper.rnd() * double(nOfActions)) %
                           nOfActions];

            goal = _goalState == _curState;

            auto& qsa = _q_mtx[_curState][nextState];
            auto& rsa = _rewardMtx[_curState][nextState];

            qsa += _learningRate *
                   (rsa + _discountRate * _q_mtx.max(nextState) - qsa);

            _curState = nextState;
        }

        helper.endEpisode(episode, *this);

        return true;
    }

    QLGraph::valid_actions_t
    QLGraph::retrieveValidActions(const QMatrix &r, size_t state)
    {
        assert(state < r.size());

        valid_actions_t va;
        const auto &actions = r[state].to_stdvec();

        size_t idx = 0;

        for (const auto &action : actions)
        {
            if (action >= 0)
            {
                va.push_back(idx);
            }
            ++idx;
        }
    }

}
