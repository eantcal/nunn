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

#include "nu_mlpnn.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <string>


/* -------------------------------------------------------------------------- */

static bool process_cl(int argc, char* argv[], std::string& load_file_name,
    std::string& save_file_name)
{
    int pidx = 1;

    for (; pidx < argc; ++pidx) {
        std::string arg = argv[pidx];

        if ((arg == "--help" || arg == "-h")) {
            return false;
        }

        if ((arg == "--version" || arg == "-v")) {
            std::cout << "nunn_topo 1.0 (c) antonino.calderone@gmail.com"
                      << std::endl;
            exit(0);
        }

        if ((arg == "--load" || arg == "-l") && (pidx + 1) < argc) {
            load_file_name = argv[++pidx];
            continue;
        }

        if ((arg == "--save" || arg == "-s") && (pidx + 1) < argc) {
            save_file_name = argv[++pidx];
            continue;
        }

        return false;
    }

    return true;
}


/* -------------------------------------------------------------------------- */

static void usage(const char* appname)
{
    std::cerr << "Usage:" << std::endl
              << appname << std::endl
              << "\t[--version|-v] " << std::endl
              << "\t[--help|-h] " << std::endl
              << "\t[--save|-s <dot file name>] " << std::endl
              << "\t[--load|-l <net_description_file_name>] " << std::endl
              << std::endl
              << "Where:" << std::endl
              << "--version or -v " << std::endl
              << "\tshows the program version" << std::endl
              << "--help or -h " << std::endl
              << "\tgenerates just this 'Usage' text " << std::endl
              << "--save or -s" << std::endl
              << "\tsave dot file" << std::endl
              << "--load or -l" << std::endl
              << "\tload net data from file" << std::endl
              << std::endl;
}


/* -------------------------------------------------------------------------- */

bool save_topo(
    const std::string& filename, nu::mlp_neural_net_t::topology_t& topology)
{
    const size_t buf_size = 1024;

    auto input_n = *topology.begin();
    const auto topology_size = topology.size();
    auto output_n = topology[topology_size - 1];
    auto hlevel_n = topology_size - 2;
    char buf[buf_size] = { 0 };

    std::stringstream ss;
    ss << "digraph G" << std::endl
       << "{" << std::endl
       << "\trankdir=LR" << std::endl
       << "\tsplines=line" << std::endl
       << "\tnodesep=.55;" << std::endl
       << "\tranksep=20;" << std::endl
       << std::endl
       << "\tnode [label=\"\", shape=circle, width=1];" << std::endl
       << std::endl
       << "\tsubgraph cluster_0 { " << std::endl
       << "\t\tcolor=white; " << std::endl
       << "\t\tnode [style=solid,color=blue4, shape=circle]; " << std::endl
       << "\t\t";

    for (size_t node = 0; node < input_n; ++node) {
        snprintf(buf, sizeof(buf) - 1, " x%03zu", node);
        ss << buf;
    }

    ss << "; " << std::endl;

    ss << "\t\tlabel = \"Input Layer\"; " << std::endl;
    ss << "\t}" << std::endl;

    for (size_t level = 1; level <= hlevel_n; ++level) {
        ss << "\tsubgraph cluster_" << level << " { " << std::endl
           << "\t\tcolor=white; " << std::endl
           << "\t\tnode [style=solid,color=red2, shape=circle]; " << std::endl
           << "\t\t";

        auto level_nodes_n = topology[level];

        for (size_t node = 0; node < level_nodes_n; ++node) {
            snprintf(buf, sizeof(buf) - 1, " a%03zu%03zu", level, node);
            ss << buf;
        }

        ss << "; " << std::endl;

        ss << "\t\tlabel = \"Hidden Layer" << level << "\"; " << std::endl;
        ss << "\t}" << std::endl;
    }

    ss << "\tsubgraph cluster_" << hlevel_n << " { " << std::endl
       << "\t\tcolor=white; " << std::endl
       << "\t\tnode [style=solid,color=green2, shape=circle]; " << std::endl
       << "\t\t";

    auto level_nodes_n = topology[hlevel_n + 1];

    for (size_t node = 0; node < level_nodes_n; ++node) {
        snprintf(buf, sizeof(buf) - 1, " y%03zu", node);
        ss << buf;
    }

    ss << "; " << std::endl;

    ss << "\t\tlabel = \"Output Layer\"; " << std::endl;
    ss << "\t}" << std::endl << std::endl;

    for (size_t node = 0; node < input_n; ++node) {
        for (size_t level_node = 0; level_node < topology[1]; ++level_node) {
            snprintf(buf, sizeof(buf) - 1, "x%03zu->a%03zu%03zu;\n", node,
                size_t(1), level_node);
            ss << buf;
        }
    }

    ss << std::endl;


    for (size_t level = 1; level < hlevel_n; ++level) {
        for (size_t level_node_l = 0; level_node_l < topology[level];
             ++level_node_l) {
            for (size_t level_node_r = 0; level_node_r < topology[level + 1];
                 ++level_node_r) {
                snprintf(buf, sizeof(buf) - 1, "a%03zu%03zu->a%03zu%03zu;\n",
                    level, level_node_l, level + 1, level_node_r);

                ss << buf;
            }
        }

        ss << std::endl;
    }

    for (size_t node = 0; node < output_n; ++node) {
        for (size_t level_node = 0; level_node < topology[topology.size() - 2];
             ++level_node) {
            snprintf(buf, sizeof(buf) - 1, "a%03zu%03zu->y%03zu;\n", hlevel_n,
                level_node, node);
            ss << buf;
        }
    }

    ss << std::endl;


    ss << "}" << std::endl;

    if (!filename.empty()) {
        std::ofstream nf(filename);
        if (nf.is_open()) {
            nf << ss.str() << std::endl;
            nf.close();
        } else {
            std::cerr << "Cannot create '" << filename << "'" << std::endl;
            return false;
        }
    } else
        std::cout << ss.str();

    return true;
}

/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
    // Parse arguments

    std::string load_file_name;
    std::string save_file_name;

    std::unique_ptr<nu::mlp_neural_net_t> net;

    if (argc > 1) {
        if (!process_cl(argc, argv, load_file_name, save_file_name)) {
            usage(argv[0]);
            return 1;
        }
    }

    if (load_file_name.empty()) {
        std::cerr << "Error: Net file name missing" << std::endl;
        return 1;
    }

    std::ifstream nf(load_file_name);
    std::stringstream ss;

    if (!nf.is_open()) {
        std::cerr << "Error: Cannot open '" << load_file_name << "'"
                  << std::endl;
        return 1;
    }

    ss << nf.rdbuf();
    nf.close();

    net = std::make_unique<nu::mlp_neural_net_t>();

    if (net == nullptr) {
        std::cerr << "Error: out of memory" << std::endl;
        return 1;
    }

    net->load(ss);

    auto topology = net->get_topology();

    if (topology.size() < 3) {
        std::cerr << "Bad topology format" << std::endl;
    }

    save_topo(save_file_name, topology);

    return 0;
}
