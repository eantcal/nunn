/*
*  This file is part of nuNN
*
*  nuNN is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  nuNN is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with nuNN; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  US
*
*  Author: <antonino.calderone@ericsson.com>, <acaldmail@gmail.com>
*
*/

/*
* Tic-Tac-Toe Demo for Windows (winttt)
* This is an interactive demo which uses a trained MLP neural network
* created by using tictactoe demo program.
* See https://sites.google.com/site/eantcal/home/c/artificial-neural-network-library
* for more information about this demo
*/


/* -------------------------------------------------------------------------- */

#include "stdafx.h"
#include "winttt.h"

#include "mnist.h"
#include "nu_mlpnn.h"

#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <thread>
#include <mutex>


/* -------------------------------------------------------------------------- */

#define PROG_VERSION "1.47"
#define ABOUT_TEXT "TicTacToe for Windows by A. Calderone (c) - 2015"
#define ABOUT_INFO "WinTTT " PROG_VERSION
#define PROG_WINXRES 664
#define PROG_WINYRES 618


#define MAX_LOADSTRING 100

#define DIGIT_SIDE_LEN 28
#define PENSIZE  10
#define CELLSIZE 5
#define GRIDSIZE (DIGIT_SIDE_LEN*CELLSIZE)
#define FRAME_SIZE 20
#define BOARDOFF_X 224
#define BOARDOFF_Y 124
#define BOARDCELLSIZE 70
#define YBMPOFF 50
#define TRAINING_TICK 1000
#define TRAINING_EPERT 100

#define FILE_FILTER "nuNN (.net)\0*.net;\0All Files (*.*)\0*.*\0\0";

#define TICTACTOE_SIDE 3
#define TICTACTOE_CELLS (TICTACTOE_SIDE*TICTACTOE_SIDE)


/* -------------------------------------------------------------------------- */

class toolbar_t
{
private:
   HWND _toolbar;
   HINSTANCE _hinstance;
   HWND _hparent;

public:
   toolbar_t(
      HWND hParentWnd,
      HINSTANCE hInstance,
      UINT idi_toolbar,
      UINT_PTR res_id,
      int n_of_bitmaps,
      TBBUTTON buttons[],
      int n_of_buttons,
      int bmwidth = 28, int bmheight = 32,
      int btwidth = 28, int btheight = 32);

   virtual void on_resize();
   virtual void on_customize();
   virtual BOOL on_notify(HWND hWnd, LPARAM lParam);

   void enable(DWORD id);
   void disable(DWORD id);

   bool get_rect(RECT& rect);
};


/* -------------------------------------------------------------------------- */

struct sample_t
{
   nu::vector_t<double> inputs;
   nu::vector_t<double> outputs;

   bool operator<( const sample_t& other ) const throw( )
   {
      return inputs<other.inputs ||
         ( inputs == other.inputs && outputs<other.outputs );
   }
};

using samples_t = std::set<sample_t>;


/* -------------------------------------------------------------------------- */

// Global Variables
static HINSTANCE hInst;								// current instance
static TCHAR szTitle[MAX_LOADSTRING];			// The title bar text
static TCHAR szWindowClass[MAX_LOADSTRING];	// the main window class name
static HFONT g_hfFont = nullptr;

static std::unique_ptr<nu::mlp_neural_net_t> g_neural_net;
static samples_t g_training_samples;

static double g_last_mse = 1.0;
static std::string g_current_file_name;
static std::string g_net_desc;
static nu::mlp_neural_net_t::topology_t g_topology({ 10, 30, 9 });
static double g_learing_rate = 0.30;
static double g_momentum = 0.50;

static std::mutex g_tsync_mtx;
static std::unique_ptr<nu::mlp_neural_net_t> g_neural_net_copy;
static samples_t g_training_samples_copy;


/* -------------------------------------------------------------------------- */

// Toolbar
static toolbar_t * g_toolbar = nullptr;
const int g_toolbar_n_of_bmps = 5;
const int g_toolbar_btn_state = TBSTATE_ENABLED;
const int g_toolbar_btn_style = BTNS_BUTTON /*| TBSTATE_ELLIPSES*/;

TBBUTTON g_toolbar_buttons[] =
{
   { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

   { 0, IDM_NEW, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "New untrained NN" },
   { 1, IDM_LOAD, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Load NN" },
   { 2, IDM_SAVE, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Save NN" },

   { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

   { 3, IDM_NEWGAME, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Restart (human)" },
   { 4, IDM_NEWGAME_SC, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Restart (computer)" },
};

const int g_toolbar_n_of_buttons = sizeof(g_toolbar_buttons) / sizeof(TBBUTTON);


/* -------------------------------------------------------------------------- */

// Forward declarations of functions included in this code module:
ATOM	MyRegisterClass(HINSTANCE hInstance);
BOOL	InitInstance(HINSTANCE, int);


LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);


/* -------------------------------------------------------------------------- */

static void realignNNCopy()
{
   g_tsync_mtx.lock();

   if ( g_neural_net )
      g_neural_net_copy =
      std::move(
      std::unique_ptr<nu::mlp_neural_net_t>(
      new nu::mlp_neural_net_t(*g_neural_net)));

   g_tsync_mtx.unlock();
}


/* -------------------------------------------------------------------------- */

int APIENTRY _tWinMain(_In_ HINSTANCE hInstance,
   _In_opt_ HINSTANCE hPrevInstance,
   _In_ LPTSTR    lpCmdLine,
   _In_ int       nCmdShow)
{
   UNREFERENCED_PARAMETER(hPrevInstance);
   UNREFERENCED_PARAMETER(lpCmdLine);

   MSG msg;
   HACCEL hAccelTable;

   // Initialize global strings
   LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
   LoadString(hInstance, IDC_WINTTT, szWindowClass, MAX_LOADSTRING);
   MyRegisterClass(hInstance);

   // Perform application initialization:
   if ( !InitInstance(hInstance, nCmdShow) )
      return FALSE;

   hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_WINTTT));

   // Main message loop:
   while ( GetMessage(&msg, NULL, 0, 0) )
   {
      if ( !TranslateAccelerator(msg.hwnd, hAccelTable, &msg) )
      {
         TranslateMessage(&msg);
         DispatchMessage(&msg);
      }
   }

   return ( int ) msg.wParam;
}


/* -------------------------------------------------------------------------- */

ATOM MyRegisterClass(HINSTANCE hInstance)
{
   WNDCLASSEX wcex;

   wcex.cbSize = sizeof(WNDCLASSEX);

   wcex.style = CS_HREDRAW | CS_VREDRAW;

   wcex.lpfnWndProc = WndProc;
   wcex.cbClsExtra = 0;
   wcex.cbWndExtra = 0;
   wcex.hInstance = hInstance;
   wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_WINTTT));
   wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
   wcex.hbrBackground = ( HBRUSH ) ( COLOR_WINDOW + 1 );
   wcex.lpszMenuName = MAKEINTRESOURCE(IDC_WINTTT);
   wcex.lpszClassName = szWindowClass;
   wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

   return RegisterClassEx(&wcex);
}


/* -------------------------------------------------------------------------- */

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   HWND hWnd;

   hInst = hInstance; // Store instance handle in our global variable

   hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);

   if ( !hWnd )
      return FALSE;

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}


/* -------------------------------------------------------------------------- */

void UpdateStatusBar(HWND hWnd)
{
   const double learning_rate = g_neural_net->get_learning_rate();
   const double momentum = g_neural_net->get_momentum();
   auto topology = g_neural_net->get_topology();

   std::string inputs;
   std::string outputs;
   std::string hl;

   for ( size_t i = 0; i<topology.size(); ++i )
   {
      if ( i == 0 )
         inputs = std::to_string(topology[i]);
      else if ( i == ( topology.size() - 1 ) )
         outputs = std::to_string(topology[i]);
      else
         hl += std::to_string(topology[i]) + " ";
   }

   const auto tsc = g_training_samples.size();
   const auto mse = g_last_mse;

   g_net_desc =
      "   LR: " + std::to_string(learning_rate) +
      "   M: " + std::to_string(momentum) +
      "   HL(" + std::to_string(topology.size() - 2) + ") :" + hl +
      "   SC: " + std::to_string(tsc) +
      "   TL: " + std::to_string(100.00-mse*100.00);

   RECT r = { 0, PROG_WINYRES - 100, PROG_WINXRES, PROG_WINYRES };
   InvalidateRect(hWnd, &r, TRUE);
   UpdateWindow(hWnd);
}


/* -------------------------------------------------------------------------- */

bool LoadNetData(HWND hWnd, HINSTANCE hInst)
{
   std::vector<char> open_file_name(MAX_PATH);
   open_file_name[0] = '\0';

   OPENFILENAME ofn = { sizeof(OPENFILENAME) };

   ofn.hwndOwner = hWnd;
   ofn.hInstance = hInst;
   ofn.lpstrFile = open_file_name.data();
   ofn.nMaxFile = ( DWORD ) open_file_name.size() - 1;

   ofn.lpstrFilter = FILE_FILTER;
   ofn.lpstrCustomFilter = 0;
   ofn.nMaxCustFilter = 0;
   ofn.nFilterIndex = 0;
   ofn.lpstrTitle = "Open File";
   ofn.Flags = OFN_HIDEREADONLY;

   if ( ::GetOpenFileName(&ofn) )
   {
      //open_document_file(open_file_name.data());

      std::ifstream nf(open_file_name.data());
      std::stringstream ss;

      if ( !nf.is_open() )
      {
         MessageBox(
            hWnd, 
            "Cannot open the file", 
            open_file_name.data(), 
            MB_ICONERROR);

         return false;
      }

      ss << nf.rdbuf();
      nf.close();

      g_current_file_name = open_file_name.data();

      try {
         g_neural_net = std::make_unique< nu::mlp_neural_net_t >(ss);
      }
      catch ( ... )
      {
         MessageBox(
            hWnd, 
            "Error loading data from file", 
            open_file_name.data(), 
            MB_ICONERROR);

         return false;
      }
   }

   if ( !g_neural_net )
      return false;

   realignNNCopy();

   SetWindowText(hWnd, g_current_file_name.c_str());
   UpdateStatusBar(hWnd);

   return true;
}


/* -------------------------------------------------------------------------- */

void Training(HWND hWnd, HWND hwndPB)
{
   if ( !g_neural_net || g_training_samples.empty() )
      return;

   static bool toggle = false;

   KillTimer(hWnd, 0);

   if ( hwndPB )
      SendMessage(hwndPB, PBM_SETPOS, ( WPARAM ) ( ( 1.0 - g_last_mse )*100.0 ), 0);

   g_tsync_mtx.lock();
   
   toggle = !toggle;

   if ( toggle )
   {
      if ( g_training_samples_copy.size() != g_training_samples.size() )
      {
         g_training_samples_copy = g_training_samples;

         g_neural_net_copy =
            std::unique_ptr<nu::mlp_neural_net_t>(new nu::mlp_neural_net_t(*g_neural_net));
      }
   }
   else
   {
      if ( g_neural_net_copy )
         g_neural_net =
            std::unique_ptr<nu::mlp_neural_net_t>(new nu::mlp_neural_net_t(*g_neural_net_copy));
   }

   g_tsync_mtx.unlock();
   
   UpdateStatusBar(hWnd);

   SetTimer(hWnd, 0, TRAINING_TICK, 0);
}



/* -------------------------------------------------------------------------- */

void TrainingThread()
{
   while ( 1 )
   {
      g_tsync_mtx.lock();

      auto & samples = g_training_samples_copy;

      if ( samples.size() < 1 || g_neural_net_copy == nullptr )
      {
         g_tsync_mtx.unlock();
         Sleep(1000);
         continue;
      }

      double err = 0;
      nu::vector_t<double> outputs;
   
      for ( int i = 0; i < TRAINING_EPERT; ++i )
      {
         for ( const auto & sample : samples )
         {
            if ( g_neural_net_copy )
            {
               g_neural_net_copy->set_inputs(sample.inputs);
               g_neural_net_copy->back_propagate(sample.outputs);
               g_neural_net_copy->get_outputs(outputs);
            }

            err +=
               nu::mlp_neural_net_t::mean_squared_error(outputs, sample.outputs);
         }

         err /= samples.size();
      }

      g_last_mse = err;

      g_tsync_mtx.unlock();
   }
}


/* -------------------------------------------------------------------------- */

void SaveNetData(HWND hWnd, HINSTANCE hInst, const std::string & filename)
{
   if ( !g_neural_net )
   {
      MessageBox(hWnd, "No neural network loaded", "Error", MB_ICONERROR);
      return;
   }

   std::stringstream ss;

   ss << *g_neural_net;

   std::ofstream nf(filename);
   if ( nf.is_open() )
   {
      nf << ss.str() << std::endl;
      nf.close();
   }
   else
   {
      MessageBox(hWnd, "Cannot save current network status", "Error", MB_ICONERROR);
      return;
   }

   g_current_file_name = filename;
   SetWindowText(hWnd, filename.c_str());
}


/* -------------------------------------------------------------------------- */

void SaveFileAs(HWND hWnd, HINSTANCE hInst)
{
   char openName[MAX_PATH] = "\0";
   strncpy(openName, g_current_file_name.c_str(), MAX_PATH - 1);

   OPENFILENAME ofn = { sizeof(ofn) };
   ofn.hwndOwner = hWnd;
   ofn.hInstance = hInst;
   ofn.lpstrFile = openName;
   ofn.nMaxFile = sizeof(openName);
   ofn.lpstrTitle = "Save File";
   ofn.lpstrFilter = FILE_FILTER;
   ofn.Flags = OFN_HIDEREADONLY;

   if ( ::GetSaveFileName(&ofn) )
      SaveNetData(hWnd, hInst, openName);
}


/* -------------------------------------------------------------------------- */

void DoSelectFont(HWND hwnd)
{
   HDC hdc = GetDC(NULL);
   LONG lfHeight = -96; 
   ReleaseDC(NULL, hdc);

   HFONT hf = CreateFont(lfHeight, 0, 0, 0, 0, TRUE, 0, 0, 0, 0, 0, 0, 0, "Verdana");

   if ( hf )
   {
      if (g_hfFont) 
         DeleteObject(g_hfFont);

      g_hfFont = hf;
   }

   
}


/* -------------------------------------------------------------------------- */

static void DrawBm(HDC hdc, HANDLE image, int x, int y)
{
   HDC hdcMem = ::CreateCompatibleDC(hdc);
   auto hbmOld = ::SelectObject(hdcMem, ( HGDIOBJ ) image);

   BITMAP bm = { 0 };
   ::GetObject(image, sizeof(bm), &bm);

   auto ret = ::BitBlt(
      hdc,
      0,
      0,
      bm.bmWidth + x,
      bm.bmHeight + y,
      hdcMem, -x, -y,
      SRCCOPY);

   ::SelectObject(hdcMem, hbmOld);
   ::DeleteDC(hdcMem);
}


/* -------------------------------------------------------------------------- */

class grid_t
{
private:
   int _grid[TICTACTOE_SIDE][TICTACTOE_SIDE];
   int _tmp_grid[TICTACTOE_SIDE][TICTACTOE_SIDE];


   inline int & _at(int x, int y) throw()
   {
      return _grid[x][y];
   }

   inline const int & _at(int x, int y) const throw( )
   {
      return _grid[x][y];
   }

   inline static int _pos2y(int pos) throw( )
   {
      return pos / TICTACTOE_SIDE;
   }

   inline static int _pos2x(int pos) throw( )
   {
      return pos % TICTACTOE_SIDE;
   }

public:
   enum symbol_t
   {
      EMPTY, X, O
   };


   grid_t()
   {
      clear();
   }

   void rotate_cw(bool invert = false)
   {
      memcpy(_tmp_grid, _grid, sizeof(_grid));

      auto _tmp = [&](int indx)
      {
         return _tmp_grid[_pos2x(indx)][_pos2y(indx)];
      };

      std::list<int> lidx = { 6, 3, 0, 7, 4, 1, 8, 5, 2 };
      std::list<int> rlidx = { 2, 5, 8, 1, 4, 7, 0, 3, 6 };

      if ( invert )
      {
         int indx = 0;
         for ( auto i : lidx )
            at(indx++) = _tmp(i);
      }
      else
      {
         int indx = 0;
         for ( auto i : rlidx )
            at(indx++) = _tmp(i);
      }
   }

   size_t size() const throw( )
   {
      return TICTACTOE_SIDE*TICTACTOE_SIDE;
   }


   int get_unique_id() const
   {
      long sum = 0;

      for ( size_t i = 0; i < size(); ++i )
         sum += long(at(int(i)) * std::pow(3, i));

      return sum;
   }


   void get_xo_cnt(int & x_cnt, int & o_cnt) const
   {
      for ( size_t i = 0; i < size(); ++i )
      {
         if ( at(int(i)) == O )
            ++o_cnt;
         else if ( at(int(i)) == X )
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
      for ( size_t i = 0; i < size(); ++i )
      {
         if ( at(int(i)) == O )
            at(int(i)) = X;
         else if ( at(int(i)) == X )
            at(int(i)) = O;
      }
   }

   friend grid_t operator-( const grid_t& g1, const grid_t& g2 )
   {
      auto result = g1;
      for ( size_t i = 0; i < g1.size(); ++i )
      {
         if ( g1.at(int(i)) == g2.at(int(i)) )
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

      return _at(x,y);
   }


   int& at(int x, int y)
   {
      assert(x >= 0 && x <= ( TICTACTOE_SIDE - 1 ));
      assert(y >= 0 && y <= ( TICTACTOE_SIDE - 1 ));

      return _at(x,y);
   }


   const int& at(int position) const
   {
      const int y = _pos2y(position);
      const int x = _pos2x(position);

      return _at(x,y);
   }

   int& at(int position)
   {
      const int y = _pos2y(position);
      const int x = _pos2x(position);

      return _at(x,y);
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

   bool get_winner_line_points(std::pair<int, int> & start, std::pair<int, int>& end) const
   {
      auto symbol = get_winner_symbol();

      if ( !symbol )
         return false;

      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
         if ( at(0, y) == symbol &&
            at(1, y) == symbol &&
            at(2, y) == symbol )
         {
            start.first = 0;
            start.second = y;
            end.first = 2;
            end.second = y;
            return true;
         }
      }

      for ( int x = 0; x < TICTACTOE_SIDE; ++x )
      {
         if ( at(x, 0) == symbol &&
            at(x, 1) == symbol &&
            at(x, 2) == symbol )
         {
            start.first = x;
            start.second = 0;
            end.first = x;
            end.second = 2;
            return true;
         }
      }

      if (
         ( at(0, 0) == symbol &&
         at(1, 1) == symbol &&
         at(2, 2) == symbol )
         )
      { 
         start.first = 0;
         start.second = 0;
         end.first = 2;
         end.second = 2;
      }

      if (
         ( at(2, 0) == symbol &&
         at(1, 1) == symbol &&
         at(0, 2) == symbol ) )
      {
         start.first = 0;
         start.second = 2;
         end.first = 2;
         end.second = 0;
         return true;
      }

      return false;
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
   const grid_t & _grid;
   HANDLE _ximg, _oimg;
public:
   renderer_t(const grid_t& grid, HANDLE ximage, HANDLE oimage)

      : 
      _grid(grid),
      _ximg(ximage),
      _oimg(oimage)
   {}

   virtual void draw(HDC hdc)
   {
      const int DX = BOARDCELLSIZE;
      const int DY = BOARDCELLSIZE;

      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
         for ( int x = 0; x < TICTACTOE_SIDE; ++x )
         {
            int symbol = _grid.at(x, y);

            switch ( symbol )
            {
               case 0:
                  break;
               case 1:
                  DrawBm(hdc, _ximg, BOARDOFF_X + x*DX, BOARDOFF_Y+y*DY);
                  break;
               case 2:
                  DrawBm(hdc, _oimg, BOARDOFF_X + x*DX, BOARDOFF_Y + y*DY);
                  break;
               default:
                  assert(0);
                  break;
            }
         }
      }

   }
};


/* -------------------------------------------------------------------------- */

class nn_io_converter_t
{
public:
   static void get_inputs(
      const grid_t & grid,
      grid_t::symbol_t turn_of_symb,
      nu::vector_t<double>& inputs)
   {
      inputs.resize(10, 0.0);
      size_t i = 0;

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

      for ( size_t i = 0; i < res.size(); ++i )
      {
         if ( res[i] != grid_t::EMPTY )
            outputs[i] = 1.0;
         else
            outputs[i] = 0.0;
      }


   }
};


/* -------------------------------------------------------------------------- */

static void ExpertAlgo(grid_t & new_grid, grid_t::symbol_t symbol)
{
   int x = 0;
   int y = 0;

   grid_t::symbol_t other_symbol = symbol == grid_t::O ? grid_t::X : grid_t::O;

   bool done = false;
  
   {
      int symcnt = 0;
      int osymcnt = 0;

      for ( int y = 0; y < TICTACTOE_SIDE; ++y )
      {
         for ( int x = 0; x < TICTACTOE_SIDE; ++x )
         {
            if ( new_grid.at(x, y) == symbol )
               ++symcnt;
            else if ( new_grid.at(x, y) == other_symbol )
               ++osymcnt;
         }
      }

      if ( symcnt == 1 && osymcnt == 2 && new_grid.at(1, 1) == symbol )
      {
         if ( ( new_grid.at(0, 0) == other_symbol &&
                new_grid.at(2, 2) == other_symbol ) ||

            ( new_grid.at(0, 2) == other_symbol &&
            new_grid.at(2, 0) == other_symbol ))
            
         {
            new_grid.at(1, 0) = symbol;
            return;
         }

         if (( new_grid.at(2) == other_symbol && 
               new_grid.at(7) == other_symbol )) 
         {
            new_grid.at(5) = symbol;
            return;
         }


         if ( ( new_grid.at(1) == other_symbol &&
            new_grid.at(8) == other_symbol ) )
         {
            new_grid.at(5) = symbol;
            return;
         }


         if ( ( new_grid.at(0) == other_symbol &&
            new_grid.at(7) == other_symbol ) )
         {
            new_grid.at(3) = symbol;
            return;
         }


         if ( ( new_grid.at(1) == other_symbol &&
            new_grid.at(6) == other_symbol ) )
         {
            new_grid.at(0) = symbol;
            return;
         }

         if ( ( new_grid.at(5) == other_symbol &&
            new_grid.at(6) == other_symbol ) )
         {
            new_grid.at(7) = symbol;
            return;
         }
      }
   }


   // Check for horizontal lines (if we can close and win)
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

   // Check for vertical lines (if we can close and win)
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


   // Check for diagonal 0,0->2,2 (if we can close and win)
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

   // Check for diagonal 2,0->0,2 (if we can close and win)
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


   // Check for horizontal lines (defend)
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

   // Check for vertical lines (defend)
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

   // Check for diagonal 0,0->2,2 (defend)
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

   // Check for diagonal 2,0->2,0 (defend)
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


   //All other default moves...

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

   // ... 
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



static void NetAnswer(
   nu::mlp_neural_net_t & nn,
   grid_t & grid,
   grid_t::symbol_t symbol)
{
   try {
   nu::vector_t<double> inputs, outputs;
   nn_io_converter_t::get_inputs(grid, symbol, inputs);

   nn.set_inputs(inputs);
   nn.feed_forward();
   nn.get_outputs(outputs);

   int i = 0;
   std::map<double, int> moves;
   for ( auto output : outputs )
      moves.insert(std::make_pair(output, i++));

   int move = 0;

   // Find best matching move (e.g. starting from higher rate move
   // check if game grid cell is empty, upon empty do the move. 
   // If cell is not empty, search for next one...)
   for ( auto it = moves.rbegin(); it != moves.rend(); ++it )
   {
      move = it->second;

      if ( grid.at(move) == grid_t::EMPTY )
      {
         grid.at(move) = symbol;
         break;
      }
   }
}
   catch ( ... )
   {
      MessageBox(0, 
         "Are you loaded an incompatibile net?", "Error", MB_ICONERROR);
   }

}


/* -------------------------------------------------------------------------- */

static void ComputerPlay(
   HWND hWnd,
   HINSTANCE hInst,
   nu::mlp_neural_net_t & nn,
   grid_t & grid,
   grid_t::symbol_t symbol)
{
   auto grid_old = grid;
   auto grid_expert_move = grid;

   // Ask to net the move
   NetAnswer(nn, grid, symbol);

   InvalidateRect(hWnd, NULL, TRUE);
   UpdateWindow(hWnd);
   
   // Ask to expert algo the move
   ExpertAlgo(grid_expert_move, symbol);
   
   // In case net answer with no expected move
   // train it with the expert algo answer
   if ( grid != grid_expert_move )
   {
      nu::vector_t<double> inputs, target;
      
      // From each example you have 4 different training samples,
      // one for each board's orientation
      for (int i = 0; i<4; ++i )
      {
         grid_old.rotate_cw();
         grid_expert_move.rotate_cw();

         nn_io_converter_t::get_inputs(grid_old, symbol, inputs);
         nn_io_converter_t::get_outputs(grid_old, grid_expert_move, target);       

         g_training_samples.insert({ inputs, target });
      }

      UpdateStatusBar(hWnd);
      
   }
}


/* -------------------------------------------------------------------------- */

static bool GetGridPos(HWND hWnd, std::pair<int, int>& gpt)
{
   POINT pt = {0};
   GetCursorPos(&pt);

   POINT sp = pt;
   ScreenToClient(hWnd, &pt);

   //pt.y -= YBMPOFF;

   if ( pt.x > BOARDOFF_X &&
      pt.y > BOARDOFF_Y &&
      pt.x < ( BOARDOFF_X + 3 * BOARDCELLSIZE ) &&
      pt.y < ( BOARDOFF_Y + 3 * BOARDCELLSIZE ) )
   {
      gpt.first = ( pt.x - BOARDOFF_X ) / BOARDCELLSIZE;
      gpt.second = ( pt.y - BOARDOFF_Y ) / BOARDCELLSIZE;

      return true;
   }

   return false;
};


/* -------------------------------------------------------------------------- */

static void NewNN(HWND hWnd)
{
   g_neural_net = 
      std::unique_ptr<nu::mlp_neural_net_t>(
      new nu::mlp_neural_net_t(g_topology, g_learing_rate, g_momentum));

   realignNNCopy();
   UpdateStatusBar(hWnd);
}


/* -------------------------------------------------------------------------- */

void RotateCW(HWND hWnd, bool acw, grid_t & grid)
{
   grid.rotate_cw(acw);
   InvalidateRect(hWnd, NULL, TRUE);
   UpdateWindow(hWnd);
}


/* -------------------------------------------------------------------------- */

static void SetNewTopology(HWND hWnd, const nu::mlp_neural_net_t::topology_t & t)
{
   g_topology = t;
   NewNN(hWnd);
}


/* -------------------------------------------------------------------------- */

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
   int wmId, wmEvent;
   PAINTSTRUCT ps;
   HDC hdc;

   static grid_t game_board;

   RECT rcClient;  // Client area of parent window.
   GetClientRect(hWnd, &rcClient);

   static int cyVScroll = GetSystemMetrics(SM_CYVSCROLL);
   static HWND hwndPB = 0;

   switch ( message )
   {
      case WM_TIMER:
         Training(hWnd, hwndPB);
         break;

      case WM_CREATE:
         InitCommonControls();
         DoSelectFont(hWnd);

         {
            LONG lStyle = GetWindowLong(hWnd, GWL_STYLE);
            lStyle &= ~( 
               WS_THICKFRAME | 
               WS_MINIMIZE | 
               WS_MAXIMIZE | 
               WS_MINIMIZEBOX | 
               WS_MAXIMIZEBOX );
            SetWindowLong(hWnd, GWL_STYLE, lStyle);

            SetWindowPos(hWnd, NULL, 0, 0, 0, 0, 
               SWP_FRAMECHANGED | 
               SWP_NOMOVE | 
               SWP_NOSIZE | 
               SWP_NOZORDER | 
               SWP_NOOWNERZORDER);
         }

         MoveWindow(hWnd, 0, 0, PROG_WINXRES, PROG_WINYRES, TRUE);

         g_toolbar = new toolbar_t(
            hWnd,
            hInst,
            IDI_TOOLBAR,
            IDI_TOOLBAR,
            g_toolbar_n_of_bmps,
            g_toolbar_buttons,
            g_toolbar_n_of_buttons);


         hwndPB = CreateWindowEx(
            0, PROGRESS_CLASS, ( LPTSTR ) NULL,
            WS_CHILD | WS_VISIBLE, 
            0,
            YBMPOFF + cyVScroll / 2,
            PROG_WINXRES,
            cyVScroll,
            hWnd, ( HMENU ) 0, hInst, NULL);

         NewNN(hWnd);
         
         SendMessage(hwndPB, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
         SetTimer(hWnd, 0, 1000, 0);
         {
            std::thread trainer(TrainingThread);
            trainer.detach();
         }
         break;

      case WM_COMMAND:
         wmId = LOWORD(wParam);
         wmEvent = HIWORD(wParam);
         // Parse the menu selections:
         switch ( wmId )
         {
            case IDM_LR_30:
            case IDM_LR_35:
            case IDM_LR_40:
               if ( g_neural_net )
               {
                  g_neural_net->set_learning_rate(
                     wmId == IDM_LR_30 ? 0.30 : ( wmId == IDM_LR_35 ? 0.35 : 0.40 ));
                  g_learing_rate = g_neural_net->get_learning_rate();
               }

               UpdateStatusBar(hWnd);
               break;

            case IDM_M_0:
            case IDM_M_30:
            case IDM_M_40:
            case IDM_M_50:
               if ( g_neural_net )
               {
                  g_neural_net->set_momentum(
                     wmId == IDM_M_0 ? 0 : ( wmId == IDM_M_30 ? 0.30 :
                     ( wmId == IDM_M_40 ? 0.40 : 0.50 ) ));
                  g_momentum = g_neural_net->get_momentum();
               }
               UpdateStatusBar(hWnd);
               break;

            case IDM_HL_30:
               SetNewTopology(hWnd, { 10, 30, 9 });
               break;

            case IDM_HL_60:
               SetNewTopology(hWnd, { 10, 60, 9 });
               break;

            case IDM_HL_90:
               SetNewTopology(hWnd, { 10, 90, 9 });
               break;

            case IDM_HL_30_30:
               SetNewTopology(hWnd, { 10, 30, 30, 9 });
               break;

            case IDM_HL_60_60:
               SetNewTopology(hWnd, { 10, 60, 60, 9 });
               break;

            case IDM_HL_90_90:
               SetNewTopology(hWnd, { 10, 90, 90, 9 });
               break;

            case IDM_NEW:
               NewNN(hWnd);
               break;

            case IDM_ROTATE_CW:
               RotateCW(hWnd, true, game_board);
               break;

            case IDM_ROTATE_ACW:
               RotateCW(hWnd, false, game_board);
               break;

            case IDM_NEWGAME:
               game_board.clear();
               InvalidateRect(hWnd, NULL, TRUE);
               break;

            case IDM_NEWGAME_SC:
               game_board.clear();

               if (g_neural_net) 
                  ComputerPlay(hWnd, hInst, *g_neural_net, game_board, grid_t::O);
               else
                  MessageBox(
                  hWnd,
                  "Please, load neural network data to proceed",
                  "Warning",
                  MB_ICONEXCLAMATION | MB_OK);

               InvalidateRect(hWnd, NULL, TRUE);
               break;

            case IDM_LOAD:
               LoadNetData(hWnd, hInst);
               break;

            case IDM_SAVE:
               SaveFileAs(hWnd, hInst);
               break;

            case IDM_EXIT:
               DestroyWindow(hWnd);
               break;

            case IDM_ABOUT:
               {
                  MessageBox(hWnd,
                     ABOUT_TEXT,
                     ABOUT_INFO,
                     MB_ICONINFORMATION | MB_OK);

               }

            default:
               return DefWindowProc(hWnd, message, wParam, lParam);
         }
         break;

      case WM_LBUTTONUP:
         {
            if ( game_board.is_completed() || 
                 game_board.get_winner_symbol() != grid_t::EMPTY  )
               break;

            if ( !g_neural_net )
            {
               MessageBox(
                  hWnd, 
                  "Please, load neural network data to proceed", 
                  "Warning",
                  MB_ICONEXCLAMATION | MB_OK);

               break;
            }

            std::pair<int,int> gtp;
            bool res = GetGridPos(hWnd, gtp);
            bool modified = false;

            if ( res )
            {
               auto & cell = game_board.at(gtp.first, gtp.second);

               switch ( cell )
               {
                  case grid_t::EMPTY:
                     cell = grid_t::X;
                     modified = true;
                     break;
                  case grid_t::X:
                  case grid_t::O:
                  default:
                     break;
               }

               InvalidateRect(hWnd, NULL, TRUE);
               UpdateWindow(hWnd);

               auto win_symb = game_board.get_winner_symbol();

               bool nnet_ok = g_neural_net != nullptr;
               
               if ( modified && nnet_ok && win_symb == grid_t::EMPTY )
               {
                  ComputerPlay(hWnd, hInst, *g_neural_net, game_board, grid_t::O);
                  win_symb = game_board.get_winner_symbol();
               }

               if ( win_symb != grid_t::EMPTY || game_board.is_completed() )
               {
                  std::pair<int, int> start;
                  std::pair<int, int> end;
                  game_board.get_winner_line_points(start, end);


                  static HPEN hpenO = CreatePen(PS_SOLID, PENSIZE, RGB(255, 0, 255));
                  static HPEN hpenX = CreatePen(PS_SOLID, PENSIZE, RGB(0, 128, 255));
                  
                  std::string verdict = "Tie";
                  HDC hdc = GetDC(hWnd);

                  switch ( win_symb )
                  {
                     case grid_t::X:
                        SelectPen(hdc, hpenX);
                        verdict = "Human player wins !";
                        break;
                     case grid_t::O:
                        SelectPen(hdc, hpenO);
                        verdict = "Neural Network player wins !";
                        break;
                     case grid_t::EMPTY:
                        break;
                  }

                  if ( win_symb )
                  {
                     MoveToEx(
                        hdc,
                        BOARDOFF_X + start.first * BOARDCELLSIZE + BOARDCELLSIZE / 2,
                        BOARDOFF_Y + start.second * BOARDCELLSIZE + BOARDCELLSIZE / 2, NULL);

                     LineTo(hdc,
                        BOARDOFF_X + end.first * BOARDCELLSIZE + BOARDCELLSIZE / 2,
                        BOARDOFF_Y + end.second * BOARDCELLSIZE + BOARDCELLSIZE / 2);

                     MessageBox(hWnd, verdict.c_str(), "Verdict", MB_ICONINFORMATION | MB_OK);
                  }

                  ReleaseDC(hWnd, hdc);
               }
            }
         }
         break;

      case WM_PAINT:
         {
            static HANDLE image = ::LoadBitmap(
               GetModuleHandle(NULL),
               MAKEINTRESOURCE(IDI_BG));

            static HANDLE image_X = ::LoadBitmap(
               GetModuleHandle(NULL),
               MAKEINTRESOURCE(IDI_BM_X));

            static HANDLE image_O = ::LoadBitmap(
               GetModuleHandle(NULL),
               MAKEINTRESOURCE(IDI_BM_O));

            renderer_t renderer(game_board, image_X, image_O);

            hdc = BeginPaint(hWnd, &ps);

            if ( image )
            {
               DrawBm(hdc, image, 0, YBMPOFF);
               renderer.draw(hdc);
            }

            if ( !g_net_desc.empty() )
            {
               RECT rcClient;  // Client area of parent window.
               GetClientRect(hWnd, &rcClient);
               rcClient.top = rcClient.bottom - GetSystemMetrics(SM_CYVSCROLL);
               FillRect(hdc, &rcClient, GetStockBrush(WHITE_BRUSH));
               TextOut(hdc, 0, rcClient.top, g_net_desc.c_str(), int(g_net_desc.size() + 1));
            }

            EndPaint(hWnd, &ps);
         }
         break;

      case WM_NOTIFY:
         if ( wParam == IDI_TOOLBAR && g_toolbar )
         {
            BOOL ret_val = g_toolbar->on_notify(hWnd, lParam);

            switch ( ( ( LPNMHDR ) lParam )->code )
            {
               case TBN_QUERYDELETE:
               case TBN_GETBUTTONINFO:
               case TBN_QUERYINSERT:
                  return ret_val;
            }
         }
         return 0;

      case WM_SIZE:
         if (g_toolbar)
            g_toolbar->on_resize();
         break;

      case WM_DESTROY:
         DestroyWindow(hwndPB);
         PostQuitMessage(0);
         break;
      default:
         return DefWindowProc(hWnd, message, wParam, lParam);
   }
   return 0;
}


/* -------------------------------------------------------------------------- */

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
   UNREFERENCED_PARAMETER(lParam);
   switch ( message )
   {
      case WM_INITDIALOG:
         return ( INT_PTR ) TRUE;

      case WM_COMMAND:
         if ( LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL )
         {
            EndDialog(hDlg, LOWORD(wParam));
            return ( INT_PTR ) TRUE;
         }
         break;
   }
   
   return ( INT_PTR ) FALSE;
}


/* -------------------------------------------------------------------------- */

toolbar_t::toolbar_t(
   HWND hWnd,
   HINSTANCE hInstance,
   UINT idi_toolbar,
   UINT_PTR res_id,
   int n_of_bitmaps,
   TBBUTTON buttons[],
   int n_of_buttons,
   int bmwidth, int bmheight,
   int btwidth, int btheight
   )
   :
   _hinstance(hInstance)
{
   _hparent = hWnd;

   _toolbar = CreateToolbarEx(
      hWnd,                  // parent
      WS_CHILD | WS_BORDER | WS_VISIBLE |
      TBSTYLE_TOOLTIPS | CCS_ADJUSTABLE | TBSTYLE_FLAT,
      idi_toolbar,           // toolbar id
      n_of_bitmaps,          // number of bitmaps
      hInstance,             // mod instance
      res_id,                // resource ID for bitmap
      0, 0,
      btwidth, btheight,     // width & height of buttons
      bmwidth, bmheight,     // width & height of bitmaps
      sizeof(TBBUTTON));     // structure size

   assert(_toolbar);

   SendMessage(
      _toolbar,
      TB_ADDBUTTONS,
      ( WPARAM ) n_of_buttons,  // number of buttons
      ( LPARAM ) buttons);
}


/* -------------------------------------------------------------------------- */

void toolbar_t::on_resize()
{
   SendMessage(_toolbar, TB_AUTOSIZE, 0L, 0L);
}


/* -------------------------------------------------------------------------- */

void toolbar_t::on_customize()
{
   // Double-click on the toolbar -- display the customization dialog.
   SendMessage(_toolbar, TB_CUSTOMIZE, 0L, 0L);
}


/* -------------------------------------------------------------------------- */

BOOL toolbar_t::on_notify(HWND hWnd, LPARAM lParam)
{
   static CHAR szBuf[128];

   LPTOOLTIPTEXT lpToolTipText;

   switch ( ( ( LPNMHDR ) lParam )->code )
   {
      case TTN_GETDISPINFO:
         // Display the ToolTip text.
         lpToolTipText = ( LPTOOLTIPTEXT ) lParam;
         lpToolTipText->lpszText = /*szBuf*/ "TODO";
         break;

      case TBN_QUERYDELETE:
         // Toolbar customization -- can we delete this button?
         return TRUE;
         break;

      case TBN_GETBUTTONINFO:
         // The toolbar needs information about a button.
         return FALSE;
         break;

      case TBN_QUERYINSERT:
         // Can this button be inserted? Just say yo.
         return TRUE;
         break;

      case TBN_CUSTHELP:
         // Need to display custom help.
         break;

      case TBN_TOOLBARCHANGE:
         on_resize();
         break;

      default:
         return TRUE;
         break;
   } // switch

   return TRUE;
}


/* -------------------------------------------------------------------------- */

void toolbar_t::enable(DWORD id)
{
   SendMessage(_toolbar, TB_ENABLEBUTTON,
      id, ( LPARAM ) MAKELONG(TRUE, 0));
}


/* -------------------------------------------------------------------------- */

void toolbar_t::disable(DWORD id)
{
   SendMessage(_toolbar, TB_ENABLEBUTTON,
      id, ( LPARAM ) MAKELONG(FALSE, 0));
}


/* -------------------------------------------------------------------------- */

bool toolbar_t::get_rect(RECT& rect)
{
   return GetWindowRect(_toolbar, &rect) != 0;
}

