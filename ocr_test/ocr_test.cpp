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


/* -------------------------------------------------------------------------- */

#include "stdafx.h"
#include "ocr_test.h"

#include "mnist.h"
#include "nu_mlpnn.h"

#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>


/* -------------------------------------------------------------------------- */

#define PROG_VERSION "1.46"
#define ABOUT_TEXT "Nunn Library and OCR Test by A. Calderone (c) - 2015"
#define ABOUT_INFO "OCR Test Version " PROG_VERSION
#define PROG_WINXRES 800
#define PROG_WINYRES 600

#define TRAINING_NET_EPOCHS 10000
#define TRAINING_NET_ERRTR  0.05
#define MAX_LOADSTRING 100

#define DIGIT_SIDE_LEN 28
#define PENSIZE  10
#define CELLSIZE 5
#define GRIDSIZE (DIGIT_SIDE_LEN*CELLSIZE)
#define FRAME_SIZE 20
#define WHITEBOARD_X 100
#define WHITEBOARD_Y 130
#define YBMPOFF 100

#define FILE_FILTER "nuNN (.net)\0*.net;\0All Files (*.*)\0*.*\0\0";



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

// Global Variables
HINSTANCE hInst;								// current instance
TCHAR szTitle[MAX_LOADSTRING];			// The title bar text
TCHAR szWindowClass[MAX_LOADSTRING];	// the main window class name


static HFONT g_hfFont = nullptr;

std::unique_ptr<nu::mlp_neural_net_t> neural_net;
std::string current_file_name;
std::string net_description = "Load a net description file (File->Load)";
nu::vector_t<double> g_hwdigit;


/* -------------------------------------------------------------------------- */

// Toolbar
static toolbar_t * g_toolbar = nullptr;
const int g_toolbar_n_of_bmps = 4;
const int g_toolbar_btn_state = TBSTATE_ENABLED;
const int g_toolbar_btn_style = BTNS_BUTTON /*| TBSTATE_ELLIPSES*/;

TBBUTTON g_toolbar_buttons[] =
{
   { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

   { 0, IDM_LOAD, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Load" },
   { 1, IDM_SAVE, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Save" },

   { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

   { 2, IDM_CLS, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Clear" },

   { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

   { 3, IDM_RECOGNIZE, g_toolbar_btn_state, g_toolbar_btn_style, { 0 }, NULL, ( INT_PTR ) "Recognize" },
};

const int g_toolbar_n_of_buttons = sizeof(g_toolbar_buttons) / sizeof(TBBUTTON);


/* -------------------------------------------------------------------------- */

// Forward declarations of functions included in this code module:
ATOM	MyRegisterClass(HINSTANCE hInstance);
BOOL	InitInstance(HINSTANCE, int);


LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);


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
   LoadString(hInstance, IDC_OCR_TEST, szWindowClass, MAX_LOADSTRING);
   MyRegisterClass(hInstance);

   // Perform application initialization:
   if ( !InitInstance(hInstance, nCmdShow) )
      return FALSE;

   hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_OCR_TEST));

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
   wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_OCR_TEST));
   wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
   wcex.hbrBackground = ( HBRUSH ) ( COLOR_WINDOW + 1 );
   wcex.lpszMenuName = MAKEINTRESOURCE(IDC_OCR_TEST);
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

      current_file_name = open_file_name.data();

      try {
         neural_net = std::make_unique< nu::mlp_neural_net_t >(ss);
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

   if (! neural_net )
      return false;

   const double learning_rate = neural_net->get_learning_rate();
   const auto & topology = neural_net->get_topology();

   std::string inputs;
   std::string outputs;
   std::string hl;

   for ( size_t i = 0; i<topology.size(); ++i )
   {
      if ( i == 0 )
         inputs = std::to_string(topology[i]);
      else if ( i == (topology.size()-1))
         outputs = std::to_string(topology[i]);
      else
         hl += std::to_string(topology[i]) + " ";
   }

   net_description =
      "   Inputs: " + inputs + 
      "   Outputs: " + outputs + 
      "   HL Neurons: " + hl;

   SetWindowText(hWnd, current_file_name.c_str());

   RECT r = { 0, PROG_WINYRES-100, PROG_WINXRES, PROG_WINYRES };
   InvalidateRect(hWnd, &r, TRUE);
   UpdateWindow(hWnd);

   return true;
}


/* -------------------------------------------------------------------------- */

void SaveNetData(HWND hWnd, HINSTANCE hInst, const std::string & filename)
{
   if ( !neural_net )
   {
      MessageBox(hWnd, "No neural network loaded", "Error", MB_ICONERROR);
      return;
   }

   std::stringstream ss;
   ss << *neural_net;

   //std::cout << ss.str() << std::endl;
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

   current_file_name = filename;
   SetWindowText(hWnd, filename.c_str());
}


/* -------------------------------------------------------------------------- */

void SaveFileAs(HWND hWnd, HINSTANCE hInst)
{
   char openName[MAX_PATH] = "\0";
   strncpy(openName, current_file_name.c_str(), MAX_PATH - 1);

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

bool TrainNet(HWND hWnd, HINSTANCE hinstance, int digit)
{
   assert( digit >= 0 && digit <= 9);

   if ( !neural_net || g_hwdigit.empty() )
      return false;

   double err = 0.0;

   RECT rcClient;  // Client area of parent window.
   GetClientRect(hWnd, &rcClient);

   int cyVScroll = GetSystemMetrics(SM_CYVSCROLL);

   HWND hwndPB = CreateWindowEx(0, PROGRESS_CLASS, ( LPTSTR ) NULL,
      WS_CHILD | WS_VISIBLE, rcClient.left,
      rcClient.bottom - cyVScroll,
      rcClient.right, cyVScroll,
      hWnd, ( HMENU ) 0, hinstance, NULL);


   const int cb = TRAINING_NET_EPOCHS;

   // Set the range and increment of the progress bar. 
   SendMessage(hwndPB, PBM_SETRANGE, 0, MAKELPARAM(0, cb));
   SendMessage(hwndPB, PBM_SETSTEP, ( WPARAM ) 1, 0);

   for ( int i = 0; i < TRAINING_NET_EPOCHS; ++i )
   {
   nu::vector_t<double> target(10, 0.0);
   target[ digit ] = 1.0;
   
   neural_net->set_inputs(g_hwdigit);
   neural_net->back_propagate(target);

      err = neural_net->mean_squared_error(target);

      SendMessage(hwndPB, PBM_STEPIT, 0, 0);

      if ( err < TRAINING_NET_ERRTR )
         break;
   }

   DestroyWindow(hwndPB);

   return true;
}


/* -------------------------------------------------------------------------- */

void GetDigitBox(int xo, int yo, HDC hdc, RECT& r)
{
   DWORD res = 0;

   r.left = GRIDSIZE + FRAME_SIZE;
   r.top = GRIDSIZE + FRAME_SIZE;
   r.right = 0;
   r.bottom = 0;

   for ( int x = FRAME_SIZE; x < ( GRIDSIZE - FRAME_SIZE/2); ++x )
   {
      for ( int y = FRAME_SIZE; y < ( GRIDSIZE - FRAME_SIZE/2); ++y )
      {
         int xcell = x + xo;
         int ycell = y + yo;

         COLORREF c = GetPixel(hdc, xcell, ycell);

         switch ( c )
         {
            case RGB(255, 255, 255):
            case RGB(0, 0, 0):
               break;
            default:
               if ( x < r.left )
                  r.left = x;

               if ( y < r.top )
                  r.top = y;

               if ( x > r.right)
                  r.right = x;

               if ( y > r.bottom )
                  r.bottom = y;

               break;
         }
      }
   }

   r.left += xo;
   r.top += yo;
   
   r.right += xo;
   r.bottom += yo;

}


/* -------------------------------------------------------------------------- */

int ReadCellValue(HDC hdc, int xo, int yo, int cell_col, int cell_row, const RECT& r)
{
   int xoff = cell_col * CELLSIZE;
   int yoff = cell_row * CELLSIZE;

   int marginx = ( GRIDSIZE - ( r.right - r.left ) ) / 2;
   xoff += (r.left - xo - marginx);

   int marginy = ( GRIDSIZE - ( r.bottom - r.top) ) / 2;
   yoff += ( r.top - yo - marginy );


   DWORD res = 0;

   if (xoff< FRAME_SIZE || xoff> (GRIDSIZE-FRAME_SIZE) ||
       yoff < FRAME_SIZE || yoff > ( GRIDSIZE - FRAME_SIZE ))
      return 0;

   for ( int x = 0; x < CELLSIZE; ++x )
   {
      for ( int y = 0; y < CELLSIZE; ++y )
      {
         int xcell = x + xo + xoff;
         int ycell = y + yo + yoff;

         COLORREF c = GetPixel( hdc, xcell, ycell );

         switch ( c )
         {
            case RGB(255,255,255):
            case RGB(0, 0, 0):
               break;
            default:
               c = c == RGB(0, 0, 255) ? RGB(255, 0, 0) : RGB(0, 0, 255);
               SetPixel(hdc, xcell, ycell, c);
               ++res;
               break;
         }
      }
   }

   return res; 
}


/* -------------------------------------------------------------------------- */

void PrintGrayscaleDigit(int xo, int yo, HDC hdc, const nu::vector_t<double>& hwdigit)
{
   size_t idx = 0;
   const int zoom = 3;

   Rectangle(hdc, 
      xo-1, 
      yo-1, 
      xo + DIGIT_SIDE_LEN*zoom+1, 
      yo+DIGIT_SIDE_LEN*zoom+1);

   for ( size_t y = 0; y < DIGIT_SIDE_LEN; ++y )
   {
      for ( size_t x = 0; x < DIGIT_SIDE_LEN; ++x )
      {
         int c = int(hwdigit[idx++] * 255);


         const auto px = int(x) * zoom + xo;
         const auto py = int(y) * zoom + yo;

         for ( auto a = 0; a<zoom; ++a )
            for ( auto b = 0; b<zoom; ++b )
               SetPixel(hdc, px + a, py + b, RGB(255 - c, 255 - c, 255 - c));
      }
      }
   }


/* -------------------------------------------------------------------------- */

bool GetDigitInfo(HDC hdc, nu::vector_t<double>& hwdigit, const RECT & r)
{
   size_t vec_idx = 0;
   double sum = 0.0;

   auto f = [](double x) { return ( x / double(CELLSIZE*CELLSIZE) ); };

   for ( int y = 0; y < GRIDSIZE / CELLSIZE; ++y )
      for ( int x = 0; x < GRIDSIZE / CELLSIZE; ++x )
      {
         const double value = 
            f(double(ReadCellValue(hdc, WHITEBOARD_X, WHITEBOARD_Y, x, y, r)));

         sum += value;

         if ( vec_idx < ( DIGIT_SIDE_LEN*DIGIT_SIDE_LEN ) )
         hwdigit[vec_idx++] = value;
      }

   return sum > 0.0;
}


/* -------------------------------------------------------------------------- */

void WriteBars(int xo, int yo, HDC hdc, nu::vector_t<double>& results)
{
   int digit = 0;

   for ( auto i = results.begin(); i != results.end(); ++i )
   {
      std::string digit_s = std::to_string( digit ++ );

      auto percent = int(*i * 100);

      std::string result_s = std::to_string(percent);

      const int step = 19;

      Rectangle(hdc, xo, digit * step + yo, xo + percent*2, digit * step + yo + 10);
   }
}



/* -------------------------------------------------------------------------- */

void RecognizeHandwrittenDigit(int xo, int yo, HWND hWnd)
{
   if ( !neural_net )
   {
      MessageBox(
         hWnd, 
         "You have to configure the neural net to complete this job", 
         "Error", 
         MB_ICONERROR);

      return;
   }

   RECT ri = { PROG_WINYRES / 2, 0, PROG_WINXRES, PROG_WINYRES };
   InvalidateRect(hWnd, &ri, TRUE);
   UpdateWindow(hWnd);

   nu::vector_t<double> hwdigit(neural_net->get_inputs_count());
   
   HDC hdc = GetDC(hWnd);

   RECT r = {0};
   GetDigitBox(xo, yo, hdc, r);

   if ( GetDigitInfo(hdc, hwdigit, r) )
   {
      int xo1 = xo + 40;
      const int yo1 = yo + 235;

      PrintGrayscaleDigit(xo1, yo1, hdc, hwdigit);

   neural_net->set_inputs(hwdigit);
   neural_net->feed_forward();

   nu::vector_t<double> outputs;
   neural_net->get_outputs(outputs);

      WriteBars(530, 90, hdc, outputs);

   int percent = int(outputs[outputs.max_item_index()] * 100);
   std::string net_answer = std::to_string(outputs.max_item_index());

   if ( percent < 1 )
      net_answer = "?";

   if ( !net_answer.empty() )
   {
      HFONT hfont_old = ( HFONT ) SelectObject(hdc, g_hfFont);

      net_answer += "             ";

         xo1 += 420;

      TextOut(
         hdc,
            xo1 + 40,
            yo1 - 40,
         net_answer.c_str(),
         int(net_answer.size() + 1));

      SelectObject(hdc, hfont_old);
      
         g_hwdigit = hwdigit;
   }

   ReleaseDC(hWnd, hdc);
   }
   else
      MessageBox(
         hWnd, 
         "Write a digit into the box", 
         "No digit found", 
         MB_ICONWARNING);
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

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
   int wmId, wmEvent;
   PAINTSTRUCT ps;
   HDC hdc;

   static int xm = 0;
   static int ym = 0;
   static bool ftime = true;
   static HCURSOR cross_hcur = LoadCursor(NULL, IDC_CROSS);
   static HCURSOR hcur = 0;

   static HBRUSH hbr = ( HBRUSH ) GetStockObject(BLACK_BRUSH);
   static HPEN hpen = CreatePen(PS_SOLID, PENSIZE, RGB(0, 0, 255));
   static HPEN hpen_white = CreatePen(PS_SOLID, PENSIZE, RGB(255, 255, 255));
   static POINT old_pt = { 0 };


   auto frame_check = [](HWND hWnd, POINT & pt)
   {
      GetCursorPos(&pt);
      ScreenToClient(hWnd, &pt);

      return ( 
         ( pt.x > WHITEBOARD_X + FRAME_SIZE && 
           pt.x <= WHITEBOARD_X + GRIDSIZE - FRAME_SIZE ) &&
         ( pt.y >= WHITEBOARD_Y + FRAME_SIZE && 
           pt.y <= WHITEBOARD_Y + GRIDSIZE - FRAME_SIZE ) );
   };

   switch ( message )
   {

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
         break;

      case WM_COMMAND:
         wmId = LOWORD(wParam);
         wmEvent = HIWORD(wParam);
         // Parse the menu selections:
         switch ( wmId )
         {
            case IDM_CLS:
               InvalidateRect(hWnd, NULL, TRUE);
               break;

            case IDM_LOAD:
               LoadNetData(hWnd, hInst);
               break;

            case IDM_SAVE:
               SaveFileAs(hWnd, hInst);
               break;

            case IDM_RECOGNIZE:
               RecognizeHandwrittenDigit(WHITEBOARD_X, WHITEBOARD_Y, hWnd);
               break;

            case IDM_EXIT:
               DestroyWindow(hWnd);
               break;

            case IDM_0:
            case IDM_1:
            case IDM_2:
            case IDM_3:
            case IDM_4:
            case IDM_5:
            case IDM_6:
            case IDM_7:
            case IDM_8:
            case IDM_9:
               if ( !TrainNet(hWnd, hInst, wmId - IDM_0) )
               {
                  MessageBox(
                     hWnd, 
                     "Cannot perform this operation", 
                     "Error", 
                     MB_ICONASTERISK);
               }
               else
               {
                  MessageBox(
                     hWnd,
                     "Save your net status if you want "
                     "to persist this training",
                     "Thank you",
                     MB_ICONINFORMATION);
               }
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

      case WM_MOUSEMOVE:

         // When moving the mouse, the user must hold down 
         // the left mouse button to draw lines. 
         if ( wParam & MK_LBUTTON )
         {
            SetCursor(cross_hcur);
            HDC hdc = GetDC(hWnd);
            
            SelectObject(hdc, hbr);
            SelectPen(hdc, hpen);
            
            POINT pt = { 0 };

            if ( frame_check(hWnd, pt) )
            {
               MoveToEx(hdc, old_pt.x, old_pt.y, NULL);
               LineTo(hdc, pt.x, pt.y);
               old_pt = pt;
            }

            ReleaseDC(hWnd, hdc);
         }
         else if ( wParam & MK_RBUTTON )
         {
            SetCursor(cross_hcur);
            HDC hdc = GetDC(hWnd);

            SelectObject(hdc, hbr);
            SelectPen(hdc, hpen_white);

            POINT pt = { 0 };

            if ( frame_check(hWnd, pt) )
               Rectangle(hdc, pt.x, pt.y, pt.x + PENSIZE, pt.y + PENSIZE);

            ReleaseDC(hWnd, hdc);
         }
         break;

      case WM_LBUTTONDOWN:
         {
            hcur = SetCursor(cross_hcur);
            HDC hdc = GetDC (hWnd);
            SelectObject(hdc, hbr);
            SelectPen(hdc, hpen);

            POINT pt = { 0 };

            if ( frame_check(hWnd, pt) )
            {
               MoveToEx(hdc, pt.x, pt.y, NULL);
               old_pt = pt;
            }

            ReleaseDC(hWnd, hdc);
         }
         break;

      case WM_RBUTTONDOWN:
         {
            hcur = SetCursor(cross_hcur);
            HDC hdc = GetDC(hWnd);
            SelectObject(hdc, hbr);
            SelectPen(hdc, hpen_white);

            POINT pt = { 0 };

            if ( frame_check(hWnd, pt) )
               Rectangle(hdc, pt.x, pt.y, pt.x + PENSIZE, pt.y + PENSIZE);

            ReleaseDC(hWnd, hdc);
         }
         break;

      case WM_LBUTTONUP:
      case WM_RBUTTONUP:
         if ( hcur )
         {
            SetCursor(hcur);
            hcur = 0;
         }
         break;

      case WM_PAINT:
         {
            static HANDLE image = ::LoadBitmap(
               GetModuleHandle(NULL),
               MAKEINTRESOURCE(IDI_BG));

         hdc = BeginPaint(hWnd, &ps);

            if ( image )
            {
               HDC hdcMem = ::CreateCompatibleDC(hdc);
               auto hbmOld = ::SelectObject(hdcMem, ( HGDIOBJ ) image);

               BITMAP bm = { 0 };
               ::GetObject(image, sizeof(bm), &bm);

               
               auto ret = ::BitBlt(
                  hdc,
                  0,
                  0,
                  bm.bmWidth,
                  bm.bmHeight + YBMPOFF,
                  hdcMem, 0, -YBMPOFF,
                  SRCCOPY);

               ::SelectObject(hdcMem, hbmOld);
               ::DeleteDC(hdcMem);
            }

         TextOut(
            hdc, 
               220,
               500,
            net_description.c_str(), 
            int(net_description.size()));

         Rectangle(hdc,
            WHITEBOARD_X + FRAME_SIZE,
            WHITEBOARD_Y + FRAME_SIZE,
               WHITEBOARD_X + GRIDSIZE - FRAME_SIZE/2,
               WHITEBOARD_Y + GRIDSIZE - FRAME_SIZE/2);

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

