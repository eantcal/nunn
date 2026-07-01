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
 *  Author: Antonino Calderone <antonino.calderone@gmail.com>
 *
 */

#include "ocr_test.h"
#include "stdafx.h"

#include "mnist.h"
#include "nu_nn_model.h"
#include "nu_mlpnn.h"
#include "nu_mlpmatrixnn.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <memory>
#include <shlobj.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define PROG_VERSION "1.56"
#define ABOUT_TEXT "OCR Test by A. Calderone (c) - 2015"
#define ABOUT_INFO "OCR Test Version " PROG_VERSION
#define PROG_WINXRES 800
#define PROG_WINYRES 600

#define TRAINING_NET_EPOCHS 10000
#define TRAINING_NET_ERRTR 0.05
#define MAX_LOADSTRING 100

#define DIGIT_SIDE_LEN 28
#define PENSIZE 10
#define CELLSIZE 5
#define GRIDSIZE (DIGIT_SIDE_LEN * CELLSIZE)
#define FRAME_SIZE 20
#define WHITEBOARD_X 100
#define WHITEBOARD_Y 130
#define YBMPOFF 100

#define FILE_FILTER "nunn JSON (.json)\0*.json;\0All Files (*.*)\0*.*\0\0";

// ── Model registry ───────────────────────────────────────────────────────────

struct ModelProfile {
    std::string name; // display name (filename stem)
    std::string path; // full path to the .json file
};


// Loading a new NN we check for the following values
#define NN_INPUTS 784
#define NN_OUTPUTS 10


class toolbar_t {
private:
    HWND _toolbar;
    HINSTANCE _hinstance;
    HWND _hparent;

public:
    toolbar_t(HWND hParentWnd, HINSTANCE hInstance, UINT idi_toolbar, UINT_PTR res_id,
        int n_of_bitmaps, TBBUTTON buttons[], int n_of_buttons, int bmwidth = 28, int bmheight = 32,
        int btwidth = 28, int btheight = 32);

    virtual void on_resize();
    virtual void on_customize();
    virtual BOOL on_notify(HWND hWnd, LPARAM lParam);

    void enable(DWORD id);
    void disable(DWORD id);

    bool get_rect(RECT& rect);
};


// Global Variables
HINSTANCE hInst; // current instance
TCHAR szTitle[MAX_LOADSTRING]; // The title bar text
TCHAR szWindowClass[MAX_LOADSTRING]; // the main window class name


static HFONT g_hfFont = nullptr;

std::unique_ptr<nu::NnModel> neuralNet;
std::string currentFileName;
std::string netDescription = "Load a net description file (File->Load or Models menu)";
std::vector<double> g_hwdigit;

static std::vector<ModelProfile> g_profiles;
static HMENU g_modelsMenu = nullptr;
static int g_activeProfile = -1;

// ── MNIST training state ──────────────────────────────────────────────────────

#define WM_TRAIN_PROGRESS (WM_APP + 10) // wParam=pct(0-100), lParam=epoch(1-based)
#define WM_TRAIN_DONE (WM_APP + 11)
#define WM_TRAIN_ERROR (WM_APP + 12) // lParam=heap-allocated char* message

static std::atomic<bool> g_trainRunning{ false };
static std::thread g_trainThread;
static std::string s_trainedModelPath; // set by WM_TRAIN_DONE handler
static std::string s_lastMnistPath;
static std::string s_lastSavePath;

struct TrainConfig {
    HWND hDlg;
    bool useMatrix;
    std::vector<size_t> hidden;
    nu::Activation activation;
    nu::CostFunction cf;
    double lr;
    double momentum;
    int epochs;
    size_t batchSize;
    std::string mnistPath;
    std::string savePath;
};


// Toolbar
static toolbar_t* gtb = nullptr;
const int gtb_n_of_bmps = 4;
const int gtb_btn_state = TBSTATE_ENABLED;
const int gtb_btn_style = BTNS_BUTTON /*| TBSTATE_ELLIPSES*/;

TBBUTTON gtb_buttons[] = {
    { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

    { 0, IDM_LOAD, gtb_btn_state, gtb_btn_style, { 0 }, NULL, (INT_PTR) "Load" },
    { 1, IDM_SAVE, gtb_btn_state, gtb_btn_style, { 0 }, NULL, (INT_PTR) "Save" },

    { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

    { 2, IDM_CLS, gtb_btn_state, gtb_btn_style, { 0 }, NULL, (INT_PTR) "Clear" },

    { 0, 0, TBSTATE_ENABLED, BTNS_SEP, { 0 }, NULL, NULL },

    { 3, IDM_RECOGNIZE, gtb_btn_state, gtb_btn_style, { 0 }, NULL, (INT_PTR) "Recognize" },
};

const int gtb_n_of_buttons = sizeof(gtb_buttons) / sizeof(TBBUTTON);


// Forward declarations of functions included in this code module:
ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK TrainDlgProc(HWND, UINT, WPARAM, LPARAM);

// ── Training helpers ──────────────────────────────────────────────────────────

static std::vector<size_t> ParseHiddenLayers(const std::string& s)
{
    std::vector<size_t> result;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            result.push_back(static_cast<size_t>(std::stoul(tok)));
        } catch (...) {
        }
    }
    return result.empty() ? std::vector<size_t>{ 300 } : result;
}

// Enable or disable all configuration controls while training is running.
static void SetTrainControlsEnabled(HWND hDlg, bool enabled)
{
    const int ids[]
        = { IDC_RADIO_MLP, IDC_RADIO_MATRIX, IDC_EDIT_HIDDEN, IDC_COMBO_ACT, IDC_RADIO_MSE,
              IDC_RADIO_CE, IDC_EDIT_LR, IDC_EDIT_MOMENTUM, IDC_EDIT_EPOCHS, IDC_EDIT_BATCH,
              IDC_EDIT_MNIST_PATH, IDC_BTN_BROWSE_MNIST, IDC_EDIT_SAVE_PATH, IDC_BTN_BROWSE_SAVE };
    for (int id : ids)
        EnableWindow(GetDlgItem(hDlg, id), enabled ? TRUE : FALSE);
}

// ── MNIST training worker (runs on a background thread) ───────────────────────

static void TrainWorkerFn(TrainConfig cfg)
{
    try {
        const std::string lblFile = cfg.mnistPath + "\\train-labels.idx1-ubyte";
        const std::string imgFile = cfg.mnistPath + "\\train-images.idx3-ubyte";

        TrainingData trainingSet(lblFile, imgFile);
        trainingSet.load();

        const auto& data = trainingSet.data();
        if (data.empty())
            throw std::runtime_error("No training samples loaded from the MNIST path");

        const size_t inputSize = data.front()->get_dx() * data.front()->get_dy();

        // Post epoch-level progress after every complete epoch.
        // The timer in the dialog proc handles sub-epoch animation.
        auto postEpochProgress = [&](int ep) {
            const int pct = (cfg.epochs > 0) ? (ep + 1) * 100 / cfg.epochs : 100;
            PostMessage(cfg.hDlg, WM_TRAIN_PROGRESS, (WPARAM)pct, (LPARAM)(ep + 1));
        };

        if (!cfg.useMatrix) {
            // ── MlpNN path ───────────────────────────────────────────────────
            std::vector<nu::MlpNN::LayerConfig> layers;
            layers.emplace_back(inputSize);
            for (size_t h : cfg.hidden)
                layers.emplace_back(h, cfg.activation);
            layers.emplace_back(size_t(10), nu::Activation::Sigmoid);

            auto net = std::make_unique<nu::MlpNN>(layers, cfg.lr, cfg.momentum, cfg.cf);

            for (int ep = 0; ep < cfg.epochs && g_trainRunning; ++ep) {
                trainingSet.reshuffle();
                for (const auto& item : trainingSet.data()) {
                    if (!g_trainRunning)
                        break;
                    nu::Vector inp, tgt;
                    item->toVect(inp);
                    item->labelToTarget(tgt);
                    net->setInputVector(inp);
                    net->backPropagate(tgt);
                }
                postEpochProgress(ep);
            }

            if (g_trainRunning && !cfg.savePath.empty()) {
                std::ofstream f(cfg.savePath);
                if (!f.is_open())
                    throw std::runtime_error("Cannot open save path: " + cfg.savePath);
                net->toJson(f);
            }
        } else {
            // ── MlpMatrixNN path ─────────────────────────────────────────────
            std::vector<nu::MlpMatrixNN::LayerConfig> layers;
            layers.emplace_back(inputSize);
            for (size_t h : cfg.hidden)
                layers.emplace_back(h, cfg.activation);
            layers.emplace_back(size_t(10), nu::Activation::Sigmoid);

            auto net = std::make_unique<nu::MlpMatrixNN>(layers, cfg.lr, cfg.momentum, cfg.cf);

            const size_t bsz = cfg.batchSize < 1 ? 1 : cfg.batchSize;

            for (int ep = 0; ep < cfg.epochs && g_trainRunning; ++ep) {
                trainingSet.reshuffle();

                if (bsz == 1) {
                    for (const auto& item : trainingSet.data()) {
                        if (!g_trainRunning)
                            break;
                        nu::Vector inp, tgt;
                        item->toVect(inp);
                        item->labelToTarget(tgt);
                        std::vector<double> inpv(inp.begin(), inp.end());
                        std::vector<double> tgtv(tgt.begin(), tgt.end());
                        net->setInputVector(inpv);
                        net->feedForward();
                        net->backPropagate(tgtv);
                    }
                } else {
                    std::vector<std::vector<double>> bIn, bTgt;
                    bIn.reserve(bsz);
                    bTgt.reserve(bsz);
                    for (const auto& item : trainingSet.data()) {
                        if (!g_trainRunning)
                            break;
                        nu::Vector inp, tgt;
                        item->toVect(inp);
                        item->labelToTarget(tgt);
                        bIn.push_back(std::vector<double>(inp.begin(), inp.end()));
                        bTgt.push_back(std::vector<double>(tgt.begin(), tgt.end()));
                        if (bIn.size() == bsz) {
                            net->trainBatch(bIn, bTgt);
                            bIn.clear();
                            bTgt.clear();
                        }
                    }
                    if (!bIn.empty() && g_trainRunning)
                        net->trainBatch(bIn, bTgt);
                }
                postEpochProgress(ep);
            }

            if (g_trainRunning && !cfg.savePath.empty()) {
                std::ofstream f(cfg.savePath);
                if (!f.is_open())
                    throw std::runtime_error("Cannot open save path: " + cfg.savePath);
                net->toJson(f);
            }
        }

        g_trainRunning = false;
        PostMessage(cfg.hDlg, WM_TRAIN_DONE, 0, 0);
    } catch (const std::exception& e) {
        g_trainRunning = false;
        char* msg = _strdup(e.what());
        PostMessage(cfg.hDlg, WM_TRAIN_ERROR, 0, reinterpret_cast<LPARAM>(msg));
    }
}

// ── Training dialog procedure ─────────────────────────────────────────────────

INT_PTR CALLBACK TrainDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    static int s_trainPct = 0; // last epoch-level percentage (0-100)
    static int s_trainEp = 0; // last completed epoch (1-based; 0 = not started)
    static bool s_blinkOn = false;

    switch (msg) {

    case WM_INITDIALOG: {
        s_trainPct = 0;
        s_trainEp = 0;
        s_blinkOn = false;
        CheckRadioButton(hDlg, IDC_RADIO_MLP, IDC_RADIO_MATRIX, IDC_RADIO_MLP);

        HWND hCombo = GetDlgItem(hDlg, IDC_COMBO_ACT);
        SendMessage(hCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Sigmoid"));
        SendMessage(hCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Tanh"));
        SendMessage(hCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("ReLU"));
        SendMessage(hCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("LeakyReLU"));
        SendMessage(hCombo, CB_SETCURSEL, 0, 0);

        CheckRadioButton(hDlg, IDC_RADIO_MSE, IDC_RADIO_CE, IDC_RADIO_MSE);

        SetDlgItemText(hDlg, IDC_EDIT_HIDDEN, "300");
        SetDlgItemText(hDlg, IDC_EDIT_LR, "0.025");
        SetDlgItemText(hDlg, IDC_EDIT_MOMENTUM, "0.5");
        SetDlgItemText(hDlg, IDC_EDIT_EPOCHS, "30");
        SetDlgItemText(hDlg, IDC_EDIT_BATCH, "1");

        if (!s_lastMnistPath.empty())
            SetDlgItemText(hDlg, IDC_EDIT_MNIST_PATH, s_lastMnistPath.c_str());
        if (!s_lastSavePath.empty())
            SetDlgItemText(hDlg, IDC_EDIT_SAVE_PATH, s_lastSavePath.c_str());

        SendDlgItemMessage(hDlg, IDC_TRAIN_PROGRESS, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        SendDlgItemMessage(hDlg, IDC_TRAIN_PROGRESS, PBM_SETPOS, 0, 0);
        return TRUE;
    }

    case WM_COMMAND: {
        const int id = LOWORD(wParam);

        if (id == IDC_BTN_BROWSE_MNIST) {
            char buf[MAX_PATH] = {};
            BROWSEINFOA bi = {};
            bi.hwndOwner = hDlg;
            bi.pszDisplayName = buf;
            bi.lpszTitle = "Select MNIST data directory";
            bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
            LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
            if (pidl) {
                char path[MAX_PATH] = {};
                SHGetPathFromIDListA(pidl, path);
                CoTaskMemFree(pidl);
                SetDlgItemText(hDlg, IDC_EDIT_MNIST_PATH, path);
            }
            return TRUE;
        }

        if (id == IDC_BTN_BROWSE_SAVE) {
            char path[MAX_PATH] = {};
            OPENFILENAMEA ofn = {};
            ofn.lStructSize = sizeof(ofn);
            ofn.hwndOwner = hDlg;
            ofn.lpstrFile = path;
            ofn.nMaxFile = MAX_PATH;
            ofn.lpstrFilter = "nunn JSON\0*.json\0All Files\0*.*\0\0";
            ofn.lpstrDefExt = "json";
            ofn.lpstrTitle = "Save Trained Network";
            ofn.Flags = OFN_OVERWRITEPROMPT | OFN_HIDEREADONLY;
            if (GetSaveFileNameA(&ofn))
                SetDlgItemText(hDlg, IDC_EDIT_SAVE_PATH, path);
            return TRUE;
        }

        if (id == IDC_BTN_TRAIN) {
            if (g_trainRunning) {
                // Become a Cancel button while training
                g_trainRunning = false;
                SetDlgItemText(hDlg, IDC_STATUS_TEXT, "Cancelling...");
                EnableWindow(GetDlgItem(hDlg, IDC_BTN_TRAIN), FALSE);
                return TRUE;
            }

            char buf[MAX_PATH] = {};

            TrainConfig cfg;
            cfg.hDlg = hDlg;
            cfg.useMatrix = (IsDlgButtonChecked(hDlg, IDC_RADIO_MATRIX) == BST_CHECKED);

            GetDlgItemText(hDlg, IDC_EDIT_HIDDEN, buf, MAX_PATH);
            cfg.hidden = ParseHiddenLayers(buf);

            static const nu::Activation kActs[] = { nu::Activation::Sigmoid, nu::Activation::Tanh,
                nu::Activation::ReLU, nu::Activation::LeakyReLU };
            int actIdx
                = static_cast<int>(SendDlgItemMessage(hDlg, IDC_COMBO_ACT, CB_GETCURSEL, 0, 0));
            cfg.activation = (actIdx >= 0 && actIdx < 4) ? kActs[actIdx] : nu::Activation::Sigmoid;

            cfg.cf = (IsDlgButtonChecked(hDlg, IDC_RADIO_CE) == BST_CHECKED)
                ? nu::CostFunction::CrossEntropy
                : nu::CostFunction::MSE;

            GetDlgItemText(hDlg, IDC_EDIT_LR, buf, sizeof(buf));
            try {
                cfg.lr = std::stod(buf);
            } catch (...) {
                cfg.lr = 0.025;
            }

            GetDlgItemText(hDlg, IDC_EDIT_MOMENTUM, buf, sizeof(buf));
            try {
                cfg.momentum = std::stod(buf);
            } catch (...) {
                cfg.momentum = 0.5;
            }

            cfg.epochs = static_cast<int>(GetDlgItemInt(hDlg, IDC_EDIT_EPOCHS, nullptr, FALSE));
            if (cfg.epochs < 1)
                cfg.epochs = 1;

            cfg.batchSize
                = static_cast<size_t>(GetDlgItemInt(hDlg, IDC_EDIT_BATCH, nullptr, FALSE));
            if (cfg.batchSize < 1)
                cfg.batchSize = 1;

            GetDlgItemText(hDlg, IDC_EDIT_MNIST_PATH, buf, MAX_PATH);
            cfg.mnistPath = buf;
            s_lastMnistPath = cfg.mnistPath;

            GetDlgItemText(hDlg, IDC_EDIT_SAVE_PATH, buf, MAX_PATH);
            cfg.savePath = buf;
            s_lastSavePath = cfg.savePath;

            if (cfg.mnistPath.empty()) {
                MessageBox(hDlg, "Please specify the MNIST data directory.", "Missing path",
                    MB_ICONWARNING);
                return TRUE;
            }

            // Start worker thread
            if (g_trainThread.joinable())
                g_trainThread.join();

            g_trainRunning = true;
            SendDlgItemMessage(hDlg, IDC_TRAIN_PROGRESS, PBM_SETPOS, 0, 0);
            SetDlgItemText(hDlg, IDC_STATUS_TEXT, "Training...");
            SetDlgItemText(hDlg, IDC_BTN_TRAIN, "Cancel");
            SetTrainControlsEnabled(hDlg, false);

            s_trainPct = 0;
            s_trainEp = 0;
            s_blinkOn = false;
            SetTimer(hDlg, 1, 500, nullptr);

            g_trainThread = std::thread(TrainWorkerFn, cfg);
            return TRUE;
        }

        if (id == IDCANCEL) {
            if (g_trainRunning) {
                if (MessageBox(hDlg, "Training is in progress. Stop it?", "Confirm",
                        MB_YESNO | MB_ICONQUESTION)
                    != IDYES)
                    return TRUE;
                g_trainRunning = false;
                KillTimer(hDlg, 1);
                SetDlgItemText(hDlg, IDC_STATUS_TEXT, "Stopping...");
                EnableWindow(GetDlgItem(hDlg, IDCANCEL), FALSE);
            }
            if (g_trainThread.joinable())
                g_trainThread.join();
            EndDialog(hDlg, IDCANCEL);
            return TRUE;
        }
        break;
    }

    case WM_TIMER: {
        if (wParam != 1)
            break;
        if (!g_trainRunning) {
            KillTimer(hDlg, 1);
            break;
        }
        // Blink the frontier: alternate bar between s_trainPct and s_trainPct-1
        s_blinkOn = !s_blinkOn;
        const int blinkPos = (s_blinkOn || s_trainPct == 0) ? s_trainPct : s_trainPct - 1;
        SendDlgItemMessage(hDlg, IDC_TRAIN_PROGRESS, PBM_SETPOS, blinkPos, 0);
        // Spinner + status text
        static const char kSpinner[] = { '-', '\\', '|', '/' };
        static int spinIdx = 0;
        spinIdx = (spinIdx + 1) % 4;
        const int totalEp = (int)GetDlgItemInt(hDlg, IDC_EDIT_EPOCHS, nullptr, FALSE);
        char buf[80];
        if (s_trainEp == 0)
            sprintf_s(buf, "%c  Training ...  (0%%)", kSpinner[spinIdx]);
        else
            sprintf_s(buf, "%c  Epoch %d / %d  (%d%%)", kSpinner[spinIdx], s_trainEp, totalEp,
                s_trainPct);
        SetDlgItemText(hDlg, IDC_STATUS_TEXT, buf);
        return TRUE;
    }

    case WM_TRAIN_PROGRESS: {
        s_trainPct = static_cast<int>(wParam); // epoch-level pct (0-100)
        s_trainEp = static_cast<int>(lParam); // completed epoch (1-based)
        SendDlgItemMessage(hDlg, IDC_TRAIN_PROGRESS, PBM_SETPOS, s_trainPct, 0);
        return TRUE;
    }

    case WM_TRAIN_DONE: {
        KillTimer(hDlg, 1);
        if (g_trainThread.joinable())
            g_trainThread.join();
        SendDlgItemMessage(hDlg, IDC_TRAIN_PROGRESS, PBM_SETPOS, 100, 0);
        SetDlgItemText(hDlg, IDC_BTN_TRAIN, "Train");
        SetTrainControlsEnabled(hDlg, true);
        EnableWindow(GetDlgItem(hDlg, IDC_BTN_TRAIN), TRUE);

        char savePath[MAX_PATH] = {};
        GetDlgItemText(hDlg, IDC_EDIT_SAVE_PATH, savePath, MAX_PATH);

        std::string doneMsg = "Training complete!";
        if (savePath[0])
            doneMsg += std::string("\nModel saved to:\n") + savePath;

        if (savePath[0]) {
            doneMsg += "\n\nLoad this model into the OCR tool now?";
            if (MessageBox(hDlg, doneMsg.c_str(), "Done", MB_YESNO | MB_ICONINFORMATION) == IDYES) {
                s_trainedModelPath = savePath;
                EndDialog(hDlg, IDOK);
                return TRUE;
            }
        } else {
            MessageBox(hDlg, doneMsg.c_str(), "Done", MB_ICONINFORMATION);
        }
        SetDlgItemText(hDlg, IDC_STATUS_TEXT, "Training complete. Ready for another run.");
        return TRUE;
    }

    case WM_TRAIN_ERROR: {
        KillTimer(hDlg, 1);
        if (g_trainThread.joinable())
            g_trainThread.join();
        char* errMsg = reinterpret_cast<char*>(lParam);
        std::string msg = "Training failed:\n";
        msg += errMsg ? errMsg : "(unknown error)";
        free(errMsg);
        MessageBox(hDlg, msg.c_str(), "Error", MB_ICONERROR);
        SetDlgItemText(hDlg, IDC_STATUS_TEXT, "Error. Check parameters and paths.");
        SetDlgItemText(hDlg, IDC_BTN_TRAIN, "Train");
        SetTrainControlsEnabled(hDlg, true);
        EnableWindow(GetDlgItem(hDlg, IDC_BTN_TRAIN), TRUE);
        return TRUE;
    }
    }
    return FALSE;
}

// Launch the training dialog; if the user accepts a trained model, load it.
static void ShowTrainMnistDialog(HWND hWnd)
{
    s_trainedModelPath.clear();
    const bool comInit = SUCCEEDED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED));
    const INT_PTR result
        = DialogBoxParam(hInst, MAKEINTRESOURCE(IDD_TRAIN_DLG), hWnd, TrainDlgProc, 0);

    if (comInit)
        CoUninitialize();

    if (result == IDOK && !s_trainedModelPath.empty()) {
        try {
            auto nn = nu::NnModel::load(s_trainedModelPath);
            if (!nn || nn->getInputSize() != NN_INPUTS || nn->getOutputSize() != NN_OUTPUTS) {
                MessageBox(hWnd,
                    "Trained model has unexpected topology (need 784 inputs, 10 outputs).",
                    "Load error", MB_ICONERROR);
                return;
            }
            neuralNet = std::move(nn);
            currentFileName = s_trainedModelPath;
            SetWindowText(hWnd, s_trainedModelPath.c_str());

            const auto topo = neuralNet->getTopology();
            std::string hl;
            for (size_t i = 1; i + 1 < topo.size(); ++i)
                hl += std::to_string(topo[i]) + (i + 2 < topo.size() ? "-" : "");
            netDescription = "[trained]  hidden: " + hl
                + "  lr: " + std::to_string(neuralNet->getLearningRate());

            RECT r = { 0, PROG_WINYRES - 100, PROG_WINXRES, PROG_WINYRES };
            InvalidateRect(hWnd, &r, TRUE);
            UpdateWindow(hWnd);
        } catch (...) {
            MessageBox(hWnd, "Failed to load trained model.", "Load error", MB_ICONERROR);
        }
    }
}


int APIENTRY _tWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPTSTR lpCmdLine, _In_ int nCmdShow)
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
    if (!InitInstance(hInstance, nCmdShow))
        return FALSE;

    hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_OCR_TEST));

    // Main message loop:
    while (GetMessage(&msg, NULL, 0, 0)) {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int)msg.wParam;
}


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
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = MAKEINTRESOURCE(IDC_OCR_TEST);
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassEx(&wcex);
}


BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    HWND hWnd;

    hInst = hInstance; // Store instance handle in our global variable

    hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0,
        CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);

    if (!hWnd)
        return FALSE;

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    return TRUE;
}


bool LoadNetData(HWND hWnd, HINSTANCE hInst)
{
    std::vector<char> open_file_name(MAX_PATH);
    open_file_name[0] = '\0';

    OPENFILENAME ofn = { sizeof(OPENFILENAME) };

    ofn.hwndOwner = hWnd;
    ofn.hInstance = hInst;
    ofn.lpstrFile = open_file_name.data();
    ofn.nMaxFile = (DWORD)open_file_name.size() - 1;

    ofn.lpstrFilter = FILE_FILTER;
    ofn.lpstrCustomFilter = 0;
    ofn.nMaxCustFilter = 0;
    ofn.nFilterIndex = 0;
    ofn.lpstrTitle = "Open File";
    ofn.Flags = OFN_HIDEREADONLY;

    if (::GetOpenFileName(&ofn)) {
        currentFileName = open_file_name.data();

        try {
            auto nn = nu::NnModel::load(currentFileName);

            if (!nn || nn->getInputSize() != NN_INPUTS || nn->getOutputSize() != NN_OUTPUTS) {
                MessageBox(hWnd,
                    "Invalid network topology detected. "
                    "It might be an invalid net status file for this application",
                    open_file_name.data(), MB_ICONERROR);

                return false;
            }

            neuralNet = std::move(nn);
        } catch (...) {
            MessageBox(hWnd, "Error loading data from file", open_file_name.data(), MB_ICONERROR);

            return false;
        }
    }

    if (!neuralNet)
        return false;

    const double learningRate = neuralNet->getLearningRate();
    const auto topology = neuralNet->getTopology();

    std::string inputs;
    std::string outputs;
    std::string hl;

    for (size_t i = 0; i < topology.size(); ++i) {
        if (i == 0)
            inputs = std::to_string(topology[i]);
        else if (i == (topology.size() - 1))
            outputs = std::to_string(topology[i]);
        else
            hl += std::to_string(topology[i]) + " ";
    }

    netDescription = "   Inputs: " + inputs + "   Outputs: " + outputs + "   HL Neurons: " + hl;

    SetWindowText(hWnd, currentFileName.c_str());

    RECT r = { 0, PROG_WINYRES - 100, PROG_WINXRES, PROG_WINYRES };
    InvalidateRect(hWnd, &r, TRUE);
    UpdateWindow(hWnd);

    return true;
}


void SaveNetData(HWND hWnd, HINSTANCE hInst, const std::string& filename)
{
    if (!neuralNet) {
        MessageBox(hWnd, "No neural network loaded", "Error", MB_ICONERROR);

        return;
    }

    std::ofstream nf(filename);
    if (!nf.is_open()) {
        MessageBox(hWnd, "Cannot save current network status", "Error", MB_ICONERROR);

        return;
    }
    neuralNet->toJson(nf);

    currentFileName = filename;
    SetWindowText(hWnd, filename.c_str());
}


void SaveFileAs(HWND hWnd, HINSTANCE hInst)
{
    char openName[MAX_PATH] = "\0";
    strncpy(openName, currentFileName.c_str(), MAX_PATH - 1);

    OPENFILENAME ofn = { sizeof(ofn) };
    ofn.hwndOwner = hWnd;
    ofn.hInstance = hInst;
    ofn.lpstrFile = openName;
    ofn.nMaxFile = sizeof(openName);
    ofn.lpstrTitle = "Save File";
    ofn.lpstrFilter = FILE_FILTER;
    ofn.Flags = OFN_HIDEREADONLY;

    if (::GetSaveFileName(&ofn))
        SaveNetData(hWnd, hInst, openName);
}


// ── Model profile helpers ─────────────────────────────────────────────────────

static std::string GetExeDir()
{
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf);
    auto pos = p.find_last_of("\\/");
    return (pos != std::string::npos) ? p.substr(0, pos) : ".";
}

static void ScanModelProfiles(const std::string& dir)
{
    std::string pattern = dir + "\\*.json";
    WIN32_FIND_DATAA fd = {};
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE)
        return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            continue;
        std::string fn(fd.cFileName);
        std::string stem = fn.substr(0, fn.rfind('.'));
        g_profiles.push_back({ stem, dir + "\\" + fn });
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static void BuildModelsMenu(HWND hWnd)
{
    HMENU mainMenu = GetMenu(hWnd);
    if (!mainMenu)
        return;
    g_modelsMenu = CreatePopupMenu();
    if (!g_modelsMenu)
        return;
    if (g_profiles.empty()) {
        AppendMenuA(g_modelsMenu, MF_STRING | MF_GRAYED, 0, "(no models found in models/)");
    } else {
        for (size_t i = 0; i < g_profiles.size(); ++i)
            AppendMenuA(
                g_modelsMenu, MF_STRING, IDM_MODEL_BASE + (UINT)i, g_profiles[i].name.c_str());
    }
    // Insert after &File (position 0) and before &Edit
    InsertMenuA(mainMenu, 1, MF_BYPOSITION | MF_POPUP, (UINT_PTR)g_modelsMenu, "&Models");
    DrawMenuBar(hWnd);
}

static void LoadModelFromProfile(HWND hWnd, int idx)
{
    if (idx < 0 || idx >= static_cast<int>(g_profiles.size()))
        return;

    const auto& prof = g_profiles[idx];

    try {
        auto nn = nu::NnModel::load(prof.path);
        if (!nn || nn->getInputSize() != NN_INPUTS || nn->getOutputSize() != NN_OUTPUTS) {
            MessageBox(hWnd, "Invalid topology (expected 784 inputs, 10 outputs)",
                prof.path.c_str(), MB_ICONERROR);
            return;
        }
        neuralNet = std::move(nn);
    } catch (...) {
        MessageBox(hWnd, "Error loading model file", prof.path.c_str(), MB_ICONERROR);
        return;
    }

    currentFileName = prof.path;
    g_activeProfile = idx;

    // Build status description
    const auto topo = neuralNet->getTopology();
    std::string hl;
    for (size_t i = 1; i + 1 < topo.size(); ++i)
        hl += std::to_string(topo[i]) + (i + 2 < topo.size() ? "-" : "");
    netDescription = "[" + prof.name + "]  hidden: " + hl
        + "  lr: " + std::to_string(neuralNet->getLearningRate());

    SetWindowText(hWnd, prof.name.c_str());

    // Update checkmark
    if (g_modelsMenu) {
        for (int j = 0; j < static_cast<int>(g_profiles.size()); ++j)
            CheckMenuItem(g_modelsMenu, IDM_MODEL_BASE + j, MF_UNCHECKED);
        CheckMenuItem(g_modelsMenu, IDM_MODEL_BASE + idx, MF_CHECKED);
    }

    RECT r = { 0, PROG_WINYRES - 100, PROG_WINXRES, PROG_WINYRES };
    InvalidateRect(hWnd, &r, TRUE);
    UpdateWindow(hWnd);
}

// ─────────────────────────────────────────────────────────────────────────────

bool TrainNet(HWND hWnd, HINSTANCE hinstance, int digit)
{
    assert(digit >= 0 && digit <= 9);

    if (!neuralNet || g_hwdigit.empty())
        return false;

    double err = 0.0;

    RECT rcClient; // Client area of parent window.
    GetClientRect(hWnd, &rcClient);

    int cyVScroll = GetSystemMetrics(SM_CYVSCROLL);

    HWND hwndPB = CreateWindowEx(0, PROGRESS_CLASS, (LPTSTR)NULL, WS_CHILD | WS_VISIBLE,
        rcClient.left, rcClient.bottom - cyVScroll, rcClient.right, cyVScroll, hWnd, (HMENU)0,
        hinstance, NULL);


    const int cb = TRAINING_NET_EPOCHS;

    // Set the range and increment of the progress bar.
    SendMessage(hwndPB, PBM_SETRANGE, 0, MAKELPARAM(0, cb));
    SendMessage(hwndPB, PBM_SETSTEP, (WPARAM)1, 0);

    for (int i = 0; i < TRAINING_NET_EPOCHS; ++i) {
        std::vector<double> target(10, 0.0);
        std::vector<double> output(10, 0.0);
        target[digit] = 1.0;

        neuralNet->setInputVector(g_hwdigit);
        neuralNet->backPropagate(target, output);

        err = neuralNet->calcMSE(target);

        SendMessage(hwndPB, PBM_STEPIT, 0, 0);

        if (err < TRAINING_NET_ERRTR)
            break;
    }

    DestroyWindow(hwndPB);

    return true;
}


class bmpImage {
private:
    std::vector<char> _data;
    int _dx = 0;
    int _dy = 0;

public:
    int getdx() const noexcept { return _dx; }

    int getdy() const noexcept { return _dy; }

    bool capture(HDC hdcWindow, HWND hWnd)
    {

        struct resource_t {
            ~resource_t()
            {
                if (hbmScreen)
                    DeleteObject(hbmScreen);
                if (hdcMemDC)
                    DeleteObject(hdcMemDC);
            }

            HDC hdcMemDC = nullptr;
            HBITMAP hbmScreen = nullptr;
        } resource;

        BITMAP bmpScreen;

        // Create a compatible DC which is used in a BitBlt from the window DC
        resource.hdcMemDC = CreateCompatibleDC(hdcWindow);

        if (!resource.hdcMemDC) {
            return false;
        }

        // Get the client area for size calculation
        RECT rcClient;
        GetClientRect(hWnd, &rcClient);

        _dx = rcClient.right - rcClient.left;
        _dy = rcClient.bottom - rcClient.top;

        // Create a compatible bitmap from the Window DC
        resource.hbmScreen = CreateCompatibleBitmap(hdcWindow, _dx, _dy);

        if (!resource.hbmScreen) {
            return false;
        }

        // Select the compatible bitmap into the compatible memory DC.
        SelectObject(resource.hdcMemDC, resource.hbmScreen);

        // Bit block transfer into our compatible memory DC.
        if (!BitBlt(resource.hdcMemDC, 0, 0, _dx, _dy, hdcWindow, 0, 0, SRCCOPY)) {
            return false;
        }

        // Get the BITMAP from the HBITMAP
        GetObject(resource.hbmScreen, sizeof(BITMAP), &bmpScreen);

        BITMAPINFOHEADER bi;

        bi.biSize = sizeof(BITMAPINFOHEADER);
        bi.biWidth = bmpScreen.bmWidth;
        bi.biHeight = bmpScreen.bmHeight;
        bi.biPlanes = 1;
        bi.biBitCount = 32;
        bi.biCompression = BI_RGB;
        bi.biSizeImage = 0;
        bi.biXPelsPerMeter = 0;
        bi.biYPelsPerMeter = 0;
        bi.biClrUsed = 0;
        bi.biClrImportant = 0;

        DWORD dwBmpSize = ((bmpScreen.bmWidth * bi.biBitCount + 31) / 32) * 4 * bmpScreen.bmHeight;

        _data.resize(dwBmpSize);

        // Gets the "bits" from the bitmap and copies them into a buffer
        // which is pointed to by _data.data().
        GetDIBits(hdcWindow, resource.hbmScreen, 0, (UINT)bmpScreen.bmHeight, _data.data(),
            (BITMAPINFO*)&bi, DIB_RGB_COLORS);

        return true;
    }


    COLORREF getPixel(int x_, int y_) const noexcept
    {

        const int x = x_; // _dx - x_ - 1;
        const int y = _dy - y_ - 1;

        const size_t off = size_t((y * _dx) + x);
        assert(off < _data.size());

        auto col = *(((COLORREF*)_data.data()) + off);

        return col;
    }
};


void GetDigitBox(int xo, int yo, HDC hdc, RECT& r, HWND hwnd, const bmpImage& image)
{
    DWORD res = 0;

    r.left = GRIDSIZE + FRAME_SIZE;
    r.top = GRIDSIZE + FRAME_SIZE;
    r.right = 0;
    r.bottom = 0;

    const int dx = (GRIDSIZE - FRAME_SIZE / 2);
    const int dy = (GRIDSIZE - FRAME_SIZE / 2);

    for (int x = FRAME_SIZE; x < dx; ++x) {
        for (int y = FRAME_SIZE; y < dy; ++y) {
            int xcell = x + xo;
            int ycell = y + yo;

            COLORREF c = image.getPixel(xcell, ycell);

            switch (c) {
            case RGB(255, 255, 255):
            case RGB(0, 0, 0):
                break;
            default:
                if (x < r.left)
                    r.left = x;

                if (y < r.top)
                    r.top = y;

                if (x > r.right)
                    r.right = x;

                if (y > r.bottom)
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


int ReadCellValue(
    HDC hdc, int xo, int yo, int cell_col, int cell_row, const RECT& r, bmpImage& image)
{
    int xoff = cell_col * CELLSIZE;
    int yoff = cell_row * CELLSIZE;

    int marginx = (GRIDSIZE - (r.right - r.left)) / 2;
    xoff += (r.left - xo - marginx);

    int marginy = (GRIDSIZE - (r.bottom - r.top)) / 2;
    yoff += (r.top - yo - marginy);


    DWORD res = 0;

    if (xoff < FRAME_SIZE || xoff > (GRIDSIZE - FRAME_SIZE) || yoff < FRAME_SIZE
        || yoff > (GRIDSIZE - FRAME_SIZE))
        return 0;

    for (int x = 0; x < CELLSIZE; ++x) {
        for (int y = 0; y < CELLSIZE; ++y) {
            int xcell = x + xo + xoff;
            int ycell = y + yo + yoff;

            COLORREF c = image.getPixel(xcell, ycell);

            switch (c) {
            case RGB(255, 255, 255):
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


void PrintGrayscaleDigit(int xo, int yo, HDC hdc, const std::vector<double>& hwdigit)
{
    size_t idx = 0;
    const int zoom = 3;

    Rectangle(hdc, xo - 1, yo - 1, xo + DIGIT_SIDE_LEN * zoom + 1, yo + DIGIT_SIDE_LEN * zoom + 1);

    for (size_t y = 0; y < DIGIT_SIDE_LEN; ++y) {
        for (size_t x = 0; x < DIGIT_SIDE_LEN; ++x) {
            int c = int(hwdigit[idx++] * 255);


            const auto px = int(x) * zoom + xo;
            const auto py = int(y) * zoom + yo;

            for (auto a = 0; a < zoom; ++a)
                for (auto b = 0; b < zoom; ++b)
                    SetPixel(hdc, px + a, py + b, RGB(255 - c, 255 - c, 255 - c));
        }
    }
}


bool GetDigitInfo(HDC hdc, std::vector<double>& hwdigit, const RECT& r, bmpImage& image)
{
    size_t vec_idx = 0;
    double sum = 0.0;

    auto f = [](double x) { return (x / double(CELLSIZE * CELLSIZE)); };

    for (int y = 0; y < GRIDSIZE / CELLSIZE; ++y)
        for (int x = 0; x < GRIDSIZE / CELLSIZE; ++x) {
            const double value
                = f(double(ReadCellValue(hdc, WHITEBOARD_X, WHITEBOARD_Y, x, y, r, image)));

            sum += value;

            if (vec_idx < (DIGIT_SIDE_LEN * DIGIT_SIDE_LEN))
                hwdigit[vec_idx++] = value;
        }

    return sum > 0.0;
}


void WriteBars(int xo, int yo, HDC hdc, std::vector<double>& results)
{
    int digit = 0;

    for (auto i = results.begin(); i != results.end(); ++i) {
        std::string digit_s = std::to_string(digit++);

        auto percent = int(*i * 100);

        std::string result_s = std::to_string(percent);

        const int step = 19;

        Rectangle(hdc, xo, digit * step + yo, xo + percent * 2, digit * step + yo + 10);
    }
}


void RecognizeHandwrittenDigit(int xo, int yo, HWND hWnd)
{
    if (!neuralNet) {
        MessageBox(hWnd, "You need to configure the neural net to complete this job", "Error",
            MB_ICONERROR);

        return;
    }

    RECT ri = { PROG_WINYRES / 2, 0, PROG_WINXRES, PROG_WINYRES };
    InvalidateRect(hWnd, &ri, TRUE);
    UpdateWindow(hWnd);

    std::vector<double> hwdigit(neuralNet->getInputSize());

    HDC hdc = GetDC(hWnd);

    bmpImage image;
    image.capture(hdc, hWnd);

    RECT r = { 0 };
    GetDigitBox(xo, yo, hdc, r, hWnd, image);

    if (GetDigitInfo(hdc, hwdigit, r, image)) {
        int xo1 = xo + 40;
        const int yo1 = yo + 235;

        PrintGrayscaleDigit(xo1, yo1, hdc, hwdigit);

        neuralNet->setInputVector(hwdigit);
        neuralNet->feedForward();

        std::vector<double> outputs;
        neuralNet->copyOutputVector(outputs);

        WriteBars(530, 90, hdc, outputs);

        const size_t bestIdx = static_cast<size_t>(
            std::max_element(outputs.begin(), outputs.end()) - outputs.begin());
        int percent = int(outputs[bestIdx] * 100);
        std::string net_answer = std::to_string(bestIdx);

        if (percent < 1)
            net_answer = "?";

        if (!net_answer.empty()) {
            HFONT hfont_old = (HFONT)SelectObject(hdc, g_hfFont);

            net_answer += " ";

            xo1 += 420;

            TextOut(hdc, xo1 + 40, yo1 - 40, net_answer.c_str(), int(net_answer.size() + 1));

            SelectObject(hdc, hfont_old);

            g_hwdigit = hwdigit;
        }
    } else
        MessageBox(hWnd, "Write a digit into the box", "No digit found", MB_ICONWARNING);

    ReleaseDC(hWnd, hdc);
}


void DoSelectFont(HWND hwnd)
{
    HDC hdc = GetDC(NULL);
    LONG lfHeight = -96;
    ReleaseDC(NULL, hdc);

    HFONT hf = CreateFont(lfHeight, 0, 0, 0, 0, TRUE, 0, 0, 0, 0, 0, 0, 0, "Verdana");

    if (hf) {
        if (g_hfFont)
            DeleteObject(g_hfFont);

        g_hfFont = hf;
    }
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    int wmId, wmEvent;
    PAINTSTRUCT ps;
    HDC hdc;

    static int xm = 0;
    static int ym = 0;
    static bool ftime = true;
    static HCURSOR hcurCross = LoadCursor(NULL, IDC_CROSS);
    static HCURSOR hcur = 0;

    static HBRUSH hbr = (HBRUSH)GetStockObject(BLACK_BRUSH);
    static HPEN hpen = CreatePen(PS_SOLID, PENSIZE, RGB(0, 0, 255));
    static HPEN hpen_white = CreatePen(PS_SOLID, PENSIZE, RGB(255, 255, 255));
    static POINT old_pt = { 0 };


    auto checkFrame = [](HWND hWnd, POINT& pt) {
        GetCursorPos(&pt);
        ScreenToClient(hWnd, &pt);

        return ((pt.x > WHITEBOARD_X + FRAME_SIZE && pt.x <= WHITEBOARD_X + GRIDSIZE - FRAME_SIZE)
            && (pt.y >= WHITEBOARD_Y + FRAME_SIZE && pt.y <= WHITEBOARD_Y + GRIDSIZE - FRAME_SIZE));
    };

    switch (message) {

    case WM_CREATE:
        InitCommonControls();

        DoSelectFont(hWnd);

        {
            LONG lStyle = GetWindowLong(hWnd, GWL_STYLE);
            lStyle
                &= ~(WS_THICKFRAME | WS_MINIMIZE | WS_MAXIMIZE | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);
            SetWindowLong(hWnd, GWL_STYLE, lStyle);

            SetWindowPos(hWnd, NULL, 0, 0, 0, 0,
                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER);
        }

        MoveWindow(hWnd, 0, 0, PROG_WINXRES, PROG_WINYRES, TRUE);

        gtb = new toolbar_t(
            hWnd, hInst, IDI_TOOLBAR, IDI_TOOLBAR, gtb_n_of_bmps, gtb_buttons, gtb_n_of_buttons);

        // Scan for pre-trained models and build the Models menu
        {
            std::string exeDir = GetExeDir();
            ScanModelProfiles(exeDir + "\\models");
            if (g_profiles.empty())
                ScanModelProfiles(exeDir); // fallback: json files next to exe
            BuildModelsMenu(hWnd);
        }
        break;

    case WM_COMMAND:
        wmId = LOWORD(wParam);
        wmEvent = HIWORD(wParam);

        // Model profile selection
        if (static_cast<UINT>(wmId) >= IDM_MODEL_BASE
            && static_cast<UINT>(wmId) < IDM_MODEL_BASE + static_cast<UINT>(g_profiles.size())) {
            LoadModelFromProfile(hWnd, static_cast<int>(wmId - IDM_MODEL_BASE));
            break;
        }

        // Parse the menu selections:
        switch (wmId) {
        case IDM_CLS:
            InvalidateRect(hWnd, NULL, TRUE);
            break;

        case IDM_LOAD:
            LoadNetData(hWnd, hInst);
            break;

        case IDM_SAVE:
            SaveFileAs(hWnd, hInst);
            break;

        case IDM_TRAIN_MNIST:
            ShowTrainMnistDialog(hWnd);
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
            if (!TrainNet(hWnd, hInst, wmId - IDM_0)) {
                MessageBox(hWnd, "Cannot perform this operation", "Error", MB_ICONASTERISK);
            } else {
                MessageBox(hWnd,
                    "Save your net status if you want "
                    "to persist this training",
                    "Thank you", MB_ICONINFORMATION);
            }
            break;

        case IDM_ABOUT: {
            MessageBox(hWnd, ABOUT_TEXT, ABOUT_INFO, MB_ICONINFORMATION | MB_OK);
        }

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
        break;

    case WM_MOUSEMOVE:

        // When moving the mouse, the user must hold down
        // the left mouse button to draw lines.
        if (wParam & MK_LBUTTON) {
            SetCursor(hcurCross);
            HDC hdc = GetDC(hWnd);

            SelectObject(hdc, hbr);
            SelectPen(hdc, hpen);

            POINT pt = { 0 };

            if (checkFrame(hWnd, pt)) {
                MoveToEx(hdc, old_pt.x, old_pt.y, NULL);
                LineTo(hdc, pt.x, pt.y);
                old_pt = pt;
            }

            ReleaseDC(hWnd, hdc);
        } else if (wParam & MK_RBUTTON) {
            SetCursor(hcurCross);
            HDC hdc = GetDC(hWnd);

            SelectObject(hdc, hbr);
            SelectPen(hdc, hpen_white);

            POINT pt = { 0 };

            if (checkFrame(hWnd, pt))
                Rectangle(hdc, pt.x, pt.y, pt.x + PENSIZE, pt.y + PENSIZE);

            ReleaseDC(hWnd, hdc);
        }
        break;

    case WM_LBUTTONDOWN: {
        hcur = SetCursor(hcurCross);
        HDC hdc = GetDC(hWnd);
        SelectObject(hdc, hbr);
        SelectPen(hdc, hpen);

        POINT pt = { 0 };

        if (checkFrame(hWnd, pt)) {
            MoveToEx(hdc, pt.x, pt.y, NULL);
            old_pt = pt;
        }

        ReleaseDC(hWnd, hdc);
    } break;

    case WM_RBUTTONDOWN: {
        hcur = SetCursor(hcurCross);
        HDC hdc = GetDC(hWnd);
        SelectObject(hdc, hbr);
        SelectPen(hdc, hpen_white);

        POINT pt = { 0 };

        if (checkFrame(hWnd, pt))
            Rectangle(hdc, pt.x, pt.y, pt.x + PENSIZE, pt.y + PENSIZE);

        ReleaseDC(hWnd, hdc);
    } break;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        if (hcur) {
            SetCursor(hcur);
            hcur = 0;
        }
        break;

    case WM_PAINT: {
        static HANDLE image = ::LoadBitmap(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_BG));

        hdc = BeginPaint(hWnd, &ps);

        if (image) {
            HDC hdcMem = ::CreateCompatibleDC(hdc);
            auto hbmOld = ::SelectObject(hdcMem, (HGDIOBJ)image);

            BITMAP bm = { 0 };
            ::GetObject(image, sizeof(bm), &bm);


            auto ret = ::BitBlt(
                hdc, 0, 0, bm.bmWidth, bm.bmHeight + YBMPOFF, hdcMem, 0, -YBMPOFF, SRCCOPY);

            ::SelectObject(hdcMem, hbmOld);
            ::DeleteDC(hdcMem);
        }

        TextOut(hdc, 220, 500, netDescription.c_str(), int(netDescription.size()));

        Rectangle(hdc, WHITEBOARD_X + FRAME_SIZE, WHITEBOARD_Y + FRAME_SIZE,
            WHITEBOARD_X + GRIDSIZE - FRAME_SIZE / 2, WHITEBOARD_Y + GRIDSIZE - FRAME_SIZE / 2);

        EndPaint(hWnd, &ps);
    } break;

    case WM_NOTIFY:
        if (wParam == IDI_TOOLBAR && gtb) {
            const auto retVal = gtb->on_notify(hWnd, lParam);

            switch (((LPNMHDR)lParam)->code) {
            case TBN_QUERYDELETE:
            case TBN_GETBUTTONINFO:
            case TBN_QUERYINSERT:
                return retVal;
            }
        }
        return 0;

    case WM_SIZE:
        if (gtb)
            gtb->on_resize();
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}


// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message) {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL) {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }

    return (INT_PTR)FALSE;
}


toolbar_t::toolbar_t(HWND hWnd, HINSTANCE hInstance, UINT idi_toolbar, UINT_PTR res_id,
    int n_of_bitmaps, TBBUTTON buttons[], int n_of_buttons, int bmwidth, int bmheight, int btwidth,
    int btheight)
    : _hinstance(hInstance)
{
    _hparent = hWnd;

    _toolbar = CreateToolbarEx(hWnd, // parent
        WS_CHILD | WS_BORDER | WS_VISIBLE | TBSTYLE_TOOLTIPS | CCS_ADJUSTABLE | TBSTYLE_FLAT,
        idi_toolbar, // toolbar id
        n_of_bitmaps, // number of bitmaps
        hInstance, // mod instance
        res_id, // resource ID for bitmap
        0, 0, btwidth, btheight, // width & height of buttons
        bmwidth, bmheight, // width & height of bitmaps
        sizeof(TBBUTTON)); // structure size

    assert(_toolbar);

    SendMessage(_toolbar, TB_ADDBUTTONS,
        (WPARAM)n_of_buttons, // number of buttons
        (LPARAM)buttons);
}


void toolbar_t::on_resize()
{
    SendMessage(_toolbar, TB_AUTOSIZE, 0L, 0L);
}


void toolbar_t::on_customize()
{
    // Double-click on the toolbar -- display the customization dialog.
    SendMessage(_toolbar, TB_CUSTOMIZE, 0L, 0L);
}


BOOL toolbar_t::on_notify(HWND hWnd, LPARAM lParam)
{
    static CHAR szBuf[128];

    LPTOOLTIPTEXT lpToolTipText;

    switch (((LPNMHDR)lParam)->code) {
    case TTN_GETDISPINFO:
        // Display the ToolTip text.
        lpToolTipText = (LPTOOLTIPTEXT)lParam;
        lpToolTipText->lpszText = const_cast<LPSTR>("TODO");
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


void toolbar_t::enable(DWORD id)
{
    SendMessage(_toolbar, TB_ENABLEBUTTON, id, (LPARAM)MAKELONG(TRUE, 0));
}


void toolbar_t::disable(DWORD id)
{
    SendMessage(_toolbar, TB_ENABLEBUTTON, id, (LPARAM)MAKELONG(FALSE, 0));
}


bool toolbar_t::get_rect(RECT& rect)
{
    return GetWindowRect(_toolbar, &rect) != 0;
}
