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
*  Author: <antonino.calderone@ericsson.com>, <acaldmail@gmail.com>
*
*/


/* -------------------------------------------------------------------------- */

#include <cassert>
#include <string>
#include <list>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <map>
#include <set>
#include <memory>

#include "nu_mlpnn.h"


/* -------------------------------------------------------------------------- */

#define TICTACTOE_SIDE 3
#define TICTACTOE_CELLS (TICTACTOE_SIDE*TICTACTOE_SIDE)
#define HIDDEN_LAYER_SIZE  30
#define LEARNING_RATE 0.30
#define MOMENTUM 0.10
#define TRAINING_EPOCH_NUMBER 100000
#define TRAINING_ERR_THRESHOLD 0.10


/* -------------------------------------------------------------------------- */

class grid_t
{
private:
   int _grid[TICTACTOE_SIDE][TICTACTOE_SIDE];
   int _tmp_grid[TICTACTOE_SIDE][TICTACTOE_SIDE];

public:
   enum symbol_t
   {
      EMPTY, X, O
   };


   grid_t()
   {
      clear();
   }


   size_t size() const throw( )
   {
      return TICTACTOE_SIDE*TICTACTOE_SIDE;
   }


   int get_unique_id() const
   {
      long sum = 0;

      for ( int i = 0; i < size(); ++i )
         sum += int(at(i) * std::pow(3, i));

      return sum;
   }


   void get_xo_cnt(int & x_cnt, int & o_cnt) const
   {
      for ( int i = 0; i < size(); ++i )
      {
         if ( at(i) == O )
            ++o_cnt;
         else if ( at(i) == X )
            ++x_cnt;
      }
   }


   bool is_equal_nofsymb() const
   {
      int x_cnt = 0;
      int o_cnt = 0;
      get_xo_cnt(x_cnt, o_cnt);

      return x_cnt == o_cnt;
   }


   int len() const
   {
      int x_cnt = 0;
      int o_cnt = 0;
      get_xo_cnt(x_cnt, o_cnt);

      return x_cnt + o_cnt;
   }


   void invert()
   {
      for ( int i = 0; i < size(); ++i )
      {
         if ( at(i) == O )
            at(i) = X;
         else if ( at(i) == X )
            at(i) = O;
      }
   }

   friend grid_t operator-( const grid_t& g1, const grid_t& g2 )
   {
      auto result = g1;
      for ( int i = 0; i < g1.size(); ++i )
      {
         if ( g1.at(i) == g2.at(i) )
            result[i] = EMPTY;
      }

      return result;
   }

   bool operator<( const grid_t& other ) const throw( )
   {
      if ( this == &other )
         return false;

      return len() < other.len() ||
         ( len() == other.len() && get_unique_id() < other.get_unique_id() );
   }


   grid_t(const grid_t&) = default;
   grid_t& operator=( const grid_t& ) = default;


   bool operator==( const grid_t& other )
   {
      return this == &other || memcmp(_grid, other._grid, sizeof(_grid)) == 0;
   }

   bool operator!=( const grid_t& other )
   {
      return !this->operator==( other );
   }

   const int& operator[](size_t i) const
   {
      return at(int(i));
   }

   int& operator[](size_t i)
   {
      return at(int(i));
   }

   void clear()
   {
      memset(_grid, 0, sizeof(_grid));
   }

   const int& at(int x, int y) const
   {
      assert(x >= 0 && x <= ( TICTACTOE_SIDE - 1 ));
      assert(y >= 0 && y <= ( TICTACTOE_SIDE - 1 ));

      return _grid[y][x];
   }


   int& at(int x, int y)
   {
      assert(x >= 0 && x <= ( TICTACTOE_SIDE - 1 ));
      assert(y >= 0 && y <= ( TICTACTOE_SIDE - 1 ));

      return _grid[y][x];
   }


   const int& at(int position) const
   {
      const int y = position / TICTACTOE_SIDE;
      const int x = position % TICTACTOE_SIDE;

      return _grid[y][x];
   }

   int& at(int position)
   {
      const int y = position / TICTACTOE_SIDE;
      const int x = position % TICTACTOE_SIDE;

      return _grid[y][x];
   }

   bool is_the_winner(const grid_t::symbol_t symbol) const
   {
      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
         if ( at(0, y) == symbol &&
            at(1, y) == symbol &&
            at(2, y) == symbol )
            return true;
      }

      for ( int x = 0; x < TICTACTOE_SIDE; ++x )
      {
         if ( at(x, 0) == symbol &&
            at(x, 1) == symbol &&
            at(x, 2) == symbol )
            return true;
      }

      return
         ( at(0, 0) == symbol &&
         at(1, 1) == symbol &&
         at(2, 2) == symbol )
         ||
         ( at(2, 0) == symbol &&
         at(1, 1) == symbol &&
         at(0, 2) == symbol );
   }

   grid_t::symbol_t get_winner_symbol() const
   {
      return
         is_the_winner(grid_t::O) ?
         grid_t::O : (
         is_the_winner(grid_t::X) ? grid_t::X : grid_t::EMPTY );
   }

   bool is_completed() const
   {
      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
         for ( int x = 0; x < TICTACTOE_SIDE; ++x )
         {
            if ( at(x, y) == grid_t::EMPTY )
               return false;
         }
      }

      return true;
   }

};



/* -------------------------------------------------------------------------- */

class renderer_t
{
private:
   std::vector< std::string > _rows;

   void _init_rows()
   {
      _rows.resize(10);
      _rows[0] = "-------------";
      _rows[1] = "|   |   |   |";
      _rows[2] = "|   |   |   |";
      _rows[3] = "|---|---|---|";
      _rows[4] = "|   |   |   |";
      _rows[5] = "|   |   |   |";
      _rows[6] = "|---|---|---|";
      _rows[7] = "|   |   |   |";
      _rows[8] = "|   |   |   |";
      _rows[9] = "-------------";
   }

public:
   virtual void draw(const grid_t& grid, bool show_nums)
   {
      char ch = '0';

      _init_rows();

      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
         for ( int x = 0; x < TICTACTOE_SIDE; ++x )
         {
            int symbol = grid.at(x, y);
            ++ch;

            auto & cell = _rows[( y + 1 ) * 3 - 1][2 + x * 4];

            switch ( symbol )
            {
               case 0:
                  cell = show_nums ? ch : ' ';
                  break;
               case 1:
                  cell = 'X';
                  break;
               case 2:
                  cell = 'O';
                  break;
               default:
                  assert(0);
                  break;
            }
         }
      }

      for ( auto row : _rows )
      {
         std::cout << row << std::endl;
      }

      std::cout << std::endl;
   }
};


/* -------------------------------------------------------------------------- */

class nn_io_converter_t
{
public:
   static
      void get_inputs(
      const grid_t & grid,
      grid_t::symbol_t turn_of_symb,
      nu::vector_t<double>& inputs)
   {
      inputs.resize(10, 0.0);
      int i = 0;

      for ( ; i < grid.size(); ++i )
      {
         const auto & item = grid[i];
         inputs[i] = 0.5*double(item);
      }

      inputs[9] = turn_of_symb == grid_t::O ? 1.0 : 0.5;
   }

   static void get_outputs(
      const grid_t & grid,
      const grid_t & new_grid,
      nu::vector_t<double>& outputs)
   {
      outputs.resize(grid.size(), 0.0);

      grid_t res = new_grid - grid;

      for ( int i = 0; i < res.size(); ++i )
      {
         if ( res[i] != grid_t::EMPTY )
            outputs[i] = 1.0;
         else
            outputs[i] = 0.0;
      }


   }
};


/* -------------------------------------------------------------------------- */

class nn_trainer_t
{
public:
   struct sample_t
   {
      nu::vector_t<double> inputs;
      nu::vector_t<double> outputs;
   };

   using samples_t = std::vector<sample_t>;

private:
   std::set<grid_t> _pos_coll;
   
   static grid_t::symbol_t _get_turn_symb(
      const grid_t & grid, grid_t::symbol_t default_symb)
   {
      int x_cnt = 0;
      int o_cnt = 0;

      for ( int i = 0; i < TICTACTOE_CELLS; ++i )
      {
         if ( grid[i] == grid_t::O )
            ++o_cnt;
         else if ( grid[i] == grid_t::X )
            ++x_cnt;
      }

      if ( x_cnt == o_cnt )
         return default_symb;

      return x_cnt > o_cnt ? grid_t::O : grid_t::X;
   }

   //Create a move using this "expert" algorithm
   static void _play(grid_t & new_grid, grid_t::symbol_t default_symb)
   {
      int x = 0;
      int y = 0;

      grid_t::symbol_t symbol = _get_turn_symb(new_grid, default_symb);
      grid_t::symbol_t other_symbol = symbol == grid_t::O ? grid_t::X : grid_t::O;

      bool done = false;

      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
         int symcnt = 0;
         int osymcnt = 0;
         int empty_pos = 0;

         for ( int x = 0; x < TICTACTOE_SIDE; ++x )
         {
            if ( new_grid.at(x, y) == symbol )
               ++symcnt;
            else if ( new_grid.at(x, y) == other_symbol )
               ++osymcnt;
            else empty_pos = x;
         }

         if ( symcnt == 2 && osymcnt == 0 )
         {
            new_grid.at(empty_pos, y) = symbol;
            return;
         }
      }

      for ( int x = 0; x < TICTACTOE_SIDE; ++x )
   {
         int symcnt = 0;
         int osymcnt = 0;
         int empty_pos = 0;

         for ( int y = 0; y < TICTACTOE_SIDE; ++y )
         {
            if ( new_grid.at(x, y) == symbol )
               ++symcnt;
            else if ( new_grid.at(x, y) == other_symbol )
               ++osymcnt;
            else empty_pos = y;
         }

         if ( symcnt == 2 && osymcnt == 0 )
         {
            new_grid.at(x, empty_pos) = symbol;
            return;
         }
      }

      int symcnt = 0;
      int osymcnt = 0;
      int empty_pos = 0;
      int d = 0;

      for ( ; d < TICTACTOE_SIDE; ++d )
      {
         if ( new_grid.at(d, d) == symbol )
            ++symcnt;
         else if ( new_grid.at(d, d) == other_symbol )
            ++osymcnt;
         else empty_pos = d;
      }

      if ( symcnt == 2 && osymcnt == 0 )
      {
         new_grid.at(empty_pos, empty_pos) = symbol;
         return;
      }

      symcnt = 0;
      osymcnt = 0;
      empty_pos = 0;
      d = 0;

      for ( ; d < TICTACTOE_SIDE; ++d )
      {
         if ( new_grid.at(2 - d, d) == symbol )
            ++symcnt;
         else if ( new_grid.at(2 - d, d) == other_symbol )
            ++osymcnt;
         else empty_pos = d;
   }

      if ( symcnt == 2 && osymcnt == 0 )
      {
         new_grid.at(2 - empty_pos, empty_pos) = symbol;
         return;
      }

      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
   {
         symcnt = 0;
         osymcnt = 0;
         empty_pos = 0;

         for ( int x = 0; x < TICTACTOE_SIDE; ++x )
         {
            if ( new_grid.at(x, y) == symbol )
               ++symcnt;
            else if ( new_grid.at(x, y) == other_symbol )
               ++osymcnt;
            else empty_pos = x;
   }

         if ( osymcnt == 2 && symcnt == 0 )
   {
            new_grid.at(empty_pos, y) = symbol;
            return;
         }
   }

      for ( int x = 0; x < TICTACTOE_SIDE; ++x )
   {
         int symcnt = 0;
         int osymcnt = 0;
         int empty_pos = 0;

         for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
            if ( new_grid.at(x, y) == symbol )
               ++symcnt;
            else if ( new_grid.at(x, y) == other_symbol )
               ++osymcnt;
            else empty_pos = y;
      }

         if ( osymcnt == 2 && symcnt == 0 )
      {
            new_grid.at(x, empty_pos) = symbol;
            return;
         }
      }

      symcnt = 0;
      osymcnt = 0;
      empty_pos = 0;
      d = 0;

      for ( ; d < TICTACTOE_SIDE; ++d )
      {
         if ( new_grid.at(d, d) == symbol )
            ++symcnt;
         else if ( new_grid.at(d, d) == other_symbol )
            ++osymcnt;
         else empty_pos = d;
   }

      if ( osymcnt == 2 && symcnt == 0 )
   {
         new_grid.at(empty_pos, empty_pos) = symbol;
         return;
   }

      symcnt = 0;
      osymcnt = 0;
      empty_pos = 0;
      d = 0;

      for ( ; d < TICTACTOE_SIDE; ++d )
      {
         if ( new_grid.at(2 - d, d) == symbol )
            ++symcnt;
         else if ( new_grid.at(2 - d, d) == other_symbol )
            ++osymcnt;
         else empty_pos = d;
      }

      if ( osymcnt == 2 && symcnt == 0 )
      {
         new_grid.at(2 - empty_pos, empty_pos) = symbol;
         return;
      }

      if ( new_grid.at(4) == grid_t::EMPTY )
      {
         new_grid.at(4) = symbol;
         return;
      }

      if ( new_grid.at(0) == grid_t::EMPTY )
      {
         new_grid.at(0) = symbol;
         return;
      }

      if ( new_grid.at(2) == grid_t::EMPTY )
{
         new_grid.at(2) = symbol;
         return;
      }

      if ( new_grid.at(6) == grid_t::EMPTY )
      {
         new_grid.at(6) = symbol;
         return;
      }

      if ( new_grid.at(8) == grid_t::EMPTY )
      {
         new_grid.at(8) = symbol;
         return;
      }

      int move = 0;
      while ( 1 )
      {
         if ( new_grid.at(move) == grid_t::EMPTY )
   {
            new_grid.at(move) = symbol;
            return;
         }

         ++move;
      }
   }

   static bool _is_invalid(grid_t & grid)
   {
      int x_cnt = 0;
      int o_cnt = 0;

      for ( int i = 0; i < grid.size(); ++i )
      {
         if ( grid[i] == grid_t::O )
            ++o_cnt;
         else  if ( grid[i] == grid_t::X )
            ++x_cnt;
         }

      return std::abs(x_cnt - o_cnt)>1 || ( ( o_cnt + x_cnt ) > 8 );
   }


   void _build_pos_coll()
         {
      grid_t grid;

      for ( int i = 0; i < 0x3FFFF; ++i )
         {
         int k = i;

         for ( int j = 0; j < TICTACTOE_CELLS; ++j )
            {
            switch ( k & 3 )
            {
               case 0:
               case 3:
                  grid.at(j) = grid_t::EMPTY;
                  break;
               case 1:
                  grid.at(j) = grid_t::X;
                  break;
               case 2:
                  grid.at(j) = grid_t::O;
                  break;
            }

            k = k >> 2;
         }

         if ( !_is_invalid(grid) )
            _pos_coll.insert(grid);
         }
      }


   void _create_sample(
      const grid_t& init_grid_st,
      grid_t::symbol_t symb_turn,
      nu::vector_t<double>& inputs,
      nu::vector_t<double>& outputs)
   {
      auto res = init_grid_st;

      _play(res, symb_turn);

      nn_io_converter_t::get_inputs(init_grid_st, symb_turn, inputs);
      nn_io_converter_t::get_outputs(init_grid_st, res, outputs);
   }

public:
   void build_training_set(samples_t & samples)
   {
      _build_pos_coll();

      renderer_t rend;

      size_t n = 0;
      for ( auto & item : _pos_coll )
   {

         int o_cnt = 0;
         int x_cnt = 0;
         item.get_xo_cnt(o_cnt, x_cnt);

         nu::vector_t<double> inputs, outputs;

         if ( o_cnt >= x_cnt )
         {
            grid_t grid = item;
            _create_sample(item, grid_t::X, inputs, outputs);

            samples.push_back({ inputs, outputs });
         }

         if ( x_cnt >= o_cnt )
         {
            grid_t grid = item;
            _create_sample(item, grid_t::O, inputs, outputs);

            samples.push_back({ inputs, outputs });
         }

         ++n;
      }

   }
};


/* -------------------------------------------------------------------------- */

class game_t
{
private:
   grid_t _grid;
   renderer_t & _renderer;
   nu::mlp_neural_net_t & _nn;
   bool _computer_alone = false;

   void _show_verdict(grid_t::symbol_t symbol, grid_t::symbol_t computer_symbol)
   {
      if ( computer_symbol == symbol )
         std::cout << "Artificial Intelligence beats Man :-)" << std::endl;

      switch ( symbol )
      {
         case grid_t::X:
            std::cout << "X wins !" << std::endl << std::endl << std::endl;            
            break;
         case grid_t::O:
            std::cout << "O wins !" << std::endl << std::endl << std::endl;
            break;
         case grid_t::EMPTY:
         default:
            std::cout << "X and O have tied the game" << std::endl;
            break;
      }
   }

public:
   game_t(
      renderer_t & renderer,
      nu::mlp_neural_net_t & nn,
      bool computer_alone = false)
      throw( ) :
      _renderer(renderer),
      _nn(nn),
      _computer_alone(computer_alone)
   {}

   void play_computer(grid_t::symbol_t symbol)
   {
      nu::vector_t<double> inputs, outputs;
      nn_io_converter_t::get_inputs(_grid, symbol, inputs);    

      _nn.set_inputs(inputs);
      _nn.feed_forward();
      _nn.get_outputs(outputs);

      int i = 0;
      std::map<double, int> moves;
      for ( auto output : outputs )
         moves.insert(std::make_pair(output, i++));

      for ( auto m : moves )
      {
         auto rate = double(int(m.first * 1000)) / 10.0;
         if (rate>0.01)
            std::cout
               << "Neuron " << m.second + 1 << " -> "
               << rate << "%" << std::endl;
      }

      int move = 0;

      // Find win move avoiding lose one it matches
      for ( auto it = moves.rbegin(); it != moves.rend(); ++it )
      {
         move = it->second;

         if ( _grid.at(move) == grid_t::EMPTY )
      {
            _grid.at(move) = symbol;
            break;
         }
      }
   }


   bool play_human(grid_t::symbol_t symbol)
   {
      std::string choice;

      std::cout
         << "Please, give me a number within the range [1.."
         << TICTACTOE_CELLS << "]: ";

      std::cin >> choice;

      int move = 0;

      try
      {
         move = std::stoi(choice);

         if ( move<1 || move>TICTACTOE_CELLS )
            return false;
      }
      catch ( ... )
      {
         return false;
      }

      --move;

      if ( _grid.at(move) != grid_t::EMPTY )
      {
         std::cout 
            << "Move not allowed, please change your choice." 
            << std::endl;

         return false;
      }

      _grid.at(move) = symbol;

      return true;
   }

   void play(bool init_flg)
   {
      grid_t::symbol_t winner = grid_t::EMPTY;
      bool completed = false;
      bool toggle = init_flg;
      bool swap_nn_player = true;

      auto comp_symb = grid_t::EMPTY;

      do
      {
         if ( toggle )
         {
            if ( !_computer_alone )
            {
               _renderer.draw(_grid, !_computer_alone);
               while ( !play_human(grid_t::X) );
            }
            else
            {
               swap_nn_player = !swap_nn_player;
               comp_symb = grid_t::X;
               play_computer(grid_t::X);
            }
         }
         else
         {
            comp_symb = grid_t::O;
            play_computer(grid_t::O);
         }

         _renderer.draw(_grid, false);

         toggle = !toggle;

         winner = _grid.get_winner_symbol();

         completed = _grid.is_completed();

      }
      while ( winner == grid_t::EMPTY && !completed );

      _renderer.draw(_grid, false);

      _show_verdict(winner, comp_symb);
   }

};


/* -------------------------------------------------------------------------- */

static void usage(const char* appname)
{
   std::cerr
      << "Usage:" << std::endl
      << appname << std::endl
      << "\t[--version|-v] " << std::endl
      << "\t[--help|-h] " << std::endl
      << "\t[--save|-s <net_description_file_name>] " << std::endl
      << "\t[--load|-l <net_description_file_name>] " << std::endl
      << "\t[--skip_training|-n] " << std::endl
      << "\t[--learning_rate|-r <rate>] " << std::endl
      << "\t[--epoch_cnt|-e <count>] " << std::endl
      << "\t[--stop_on_err_tr|-x <error rate>] " << std::endl
      << "\t[[--hidden_layer|-hl <size> [--hidden_layer|--hl <size] ... ]  " 
      << std::endl
      << std::endl
      << "Where:" << std::endl
      << "--version or -v " << std::endl
      << "\tshows the program version" << std::endl
      << "--help or -h " << std::endl
      << "\tgenerates just this 'Usage' text " << std::endl
      << "--save or -s" << std::endl
      << "\tsave net data to file" << std::endl
      << "--load or -l" << std::endl
      << "\tload net data from file" << std::endl
      << "--skip_training or -n" << std::endl
      << "\tskip net training" << std::endl
      << "--learning_rate or -r" << std::endl
      << "\tset learning rate (default " << LEARNING_RATE << ")" << std::endl
      << "--epoch_cnt or -e" << std::endl
      << "\tset epoch count (default " << TRAINING_EPOCH_NUMBER << ")" << std::endl
      << "--stop_on_err_tr or -x" << std::endl
      << "\tset error rate threshold (default " << TRAINING_ERR_THRESHOLD << ")" 
      << std::endl
      << "--hidden_layer or -hl" << std::endl
      << "\tset hidden layer size (n. of neurons, default " << HIDDEN_LAYER_SIZE 
      << " )" << std::endl;
}


/* -------------------------------------------------------------------------- */

static bool process_cl(
   int argc,
   char* argv[],
   std::string & files_path,
   std::string & load_file_name,
   std::string & save_file_name,
   bool & skip_training,
   double & learning_rate,
   bool & change_lr,
   double & momentum,
   bool & change_m,
   int & epoch,
   double& threshold,
   std::vector<size_t>& hidden_layer
   )
{
   int pidx = 1;

   for ( ; pidx < argc; ++pidx )
   {
      std::string arg = argv[pidx];

      if (
         ( arg == "--help" || arg == "-h" ) )
      {
         return false;
      }

      if (
         ( arg == "--version" || arg == "-v" ) )
      {
         std::cout
            << "nunnlib TicTacToe 1.0 (c) acaldmail@gmail.com"
            << std::endl;
         continue;
      }

      if (
         ( arg == "--training_files_path" || arg == "-p" ) &&
         ( pidx + 1 ) < argc )
      {
         files_path = argv[++pidx];

         if ( !files_path.empty() )
         {
            if ( files_path.c_str()[files_path.size() - 1] != '/' )
               files_path += "/";
         }

         continue;
      }

      if (
         ( arg == "--skip_training" || arg == "-n" ) )
      {
         skip_training = true;
         continue;
   }

      if ( ( arg == "--load" || arg == "-l" ) &&
         ( pidx + 1 ) < argc )
      {
         load_file_name = argv[++pidx];
         continue;
      }

      if ( ( arg == "--save" || arg == "-s" ) &&
         ( pidx + 1 ) < argc )
   {
         save_file_name = argv[++pidx];
         continue;
      }

      if ( ( arg == "--learning_rate" || arg == "-r" ) &&
         ( pidx + 1 ) < argc )
      {
         try {
            learning_rate = std::stod(argv[++pidx]);
            change_lr = true;
         }
         catch ( ... )
         {
            return false;
         }
         continue;
      }

      if ( ( arg == "--momentun" || arg == "-m" ) &&
         ( pidx + 1 ) < argc )
      {
         try {
            momentum = std::stod(argv[++pidx]);
            change_m = true;
         }
         catch ( ... )
         {
            return false;
         }
         continue;
      }


      if ( ( arg == "--epoch_num" || arg == "-e" ) &&
         ( pidx + 1 ) < argc )
      {
         try {
            epoch = std::stoi(argv[++pidx]);
         }
         catch ( ... )
         {
            return false;
         }
         continue;
      }

      if ( ( arg == "--stop_on_err_tr" || arg == "-x" ) &&
         ( pidx + 1 ) < argc )
      {
         try {
            threshold = std::stod(argv[++pidx]);
         }
         catch ( ... )
         {
            return false;
         }
         continue;
      }


      if ( ( arg == "--hidden_layer" || arg == "-hl" ) &&
         ( pidx + 1 ) < argc )
      {
         try {
            hidden_layer.push_back(std::stoi(argv[++pidx]));
         }
         catch ( ... )
         {
            return false;
         }
         continue;
      }

      return false;
   }

   return true;
         }


/* -------------------------------------------------------------------------- */

bool save_the_net(const std::string& filename, std::stringstream & ss)
{
   // Save the net status if needed //

   std::ofstream nf(filename);
   if ( nf.is_open() )
   {
      nf << ss.str() << std::endl;
      nf.close();
   }
   else
   {
      std::cerr << "Cannot open '" << filename << "'" << std::endl;
      return false;
      }

   return true;
   }


/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
   renderer_t renderer;

   // Parse arguments
   std::string files_path;
   std::string load_file_name;
   std::string save_file_name;
   bool save_to_file = false;
   bool skip_training = false;
   double learning_rate = LEARNING_RATE;
   double momentum = MOMENTUM;
   int epoch_cnt = TRAINING_EPOCH_NUMBER;

   std::vector<size_t> hidden_layer;

   bool change_lr = false;
   bool change_m = false;

   double threshold = TRAINING_ERR_THRESHOLD;

   if ( argc > 1 )
   {
      if ( !process_cl(
         argc, argv,
         files_path,
         load_file_name,
         save_file_name,
         skip_training,
         learning_rate,
         change_lr,
         momentum,
         change_m,
         epoch_cnt,
         threshold,
         hidden_layer) )
      {
         usage(argv[0]);
         return 1;
      }
   }

   if ( hidden_layer.empty() )
      hidden_layer.push_back(HIDDEN_LAYER_SIZE);

   int hl_cnt = 0;
   for ( const auto & hl : hidden_layer )
   {
      std::cout
         << "NN hidden neurons L"
         << hl_cnt + 1;

      std::cout
         << "       : "
         << hidden_layer[hl_cnt++] << std::endl;
   }

   std::cout
      << "Net Learning rate  ( LR )  : " << learning_rate << std::endl;

   std::cout
      << "Net Momentum       ( M )   : " << momentum << std::endl;

   std::cout
      << "MSE Threshold      ( T )   : " << threshold << std::endl;


   std::unique_ptr<nu::mlp_neural_net_t> net;

   if ( !skip_training )
   {
      // Set up the topology
      nu::mlp_neural_net_t::topology_t topology;

      topology.push_back(TICTACTOE_CELLS + 1);

      for ( auto hl : hidden_layer )
         topology.push_back(hl);

      topology.push_back(TICTACTOE_CELLS /*outputs*/);

      net = std::unique_ptr<nu::mlp_neural_net_t>(
         new nu::mlp_neural_net_t(topology, learning_rate));
   }

   if ( !load_file_name.empty() )
   {
      std::ifstream nf(load_file_name);
      std::stringstream ss;

      if ( !nf.is_open() )
      {
         std::cerr << "Cannot open '" << load_file_name << "'" << std::endl;
         return 1;
      }

      ss << nf.rdbuf();
      nf.close();

      net = std::unique_ptr<nu::mlp_neural_net_t>(new nu::mlp_neural_net_t(ss));
   }

   if ( net == nullptr )
   {
      std::cerr
         << "Error: net not initialized... change parameters and retry"
         << std::endl;
      return 1;
   }

   if ( change_lr )
      net->set_learning_rate(learning_rate);

   if ( change_m )
      net->set_momentum(momentum);

   size_t cnt = 0;

   const int max_epoch_number = epoch_cnt;
   double best_performance = 100.0;
   int best_epoch = 0;

   if ( !skip_training )
   {
      std::cout << "Creating training set... ";
      nn_trainer_t::samples_t samples;
      nn_trainer_t trainer;
      trainer.build_training_set(samples);
      std::cout << "done." << std::endl;
      std::cout << std::endl;

      for ( int epoch = 0; epoch < max_epoch_number; ++epoch )
      {
         double mean_squared_error = 0.0;

         std::cout
            << "Learning epoch " << epoch + 1
            << " of " << max_epoch_number
            << " ( LR = " << net->get_learing_rate()
            << ", M = " << net->get_momentum()
            << ", T = " << threshold
            << " )"
            
            << std::endl
            << std::endl;

         cnt = 0;

         double min_err = 1.0;

         int max_epoch = 200000;
         double err = 0.0;

         for ( const auto & sample : samples )
         {
            auto & target = sample.outputs;
            nu::vector_t<double> outputs;

            net->set_inputs(sample.inputs);
            net->back_propagate(target);
            net->get_outputs(outputs);

            err += nu::mlp_neural_net_t::mean_squared_error(outputs, target);
         }

         double mean_err = err / samples.size();

         std::cout << "MSE=" << mean_err << std::endl;

         if ( mean_err < min_err )
         {
            min_err = mean_err;

            if ( !save_file_name.empty() )
            {
               std::stringstream ss;
               net->save(ss);
               save_the_net(save_file_name, ss);
            }

            if ( mean_err < threshold )
               break;
         }
      }
   }



   while ( 1 )
   {
      bool turn_for_beginning = true;

      while ( 1 )
      {
         turn_for_beginning = !turn_for_beginning;
         game_t game(renderer, *net);
         game.play(turn_for_beginning);

         std::string what;

         while ( what != "c" && what != "q" )
         {
            std::cout
               << "Press 'c'-continue, "
               << "'q'-quit and ENTER to confirm"
               << std::endl;

            std::cin >> what;
         }

         if ( what == "q" )
            return 0;

         if ( what == "c" )
            continue;
      }


   }

   return 0;
}

