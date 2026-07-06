//
// This file is part of the nunn Library
// Copyright (c) 2026 Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#ifndef NOMINMAX
#define NOMINMAX
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::wstring parentDir(const std::wstring& path)
{
    const auto pos = path.find_last_of(L"\\/");
    return pos == std::wstring::npos ? L"." : path.substr(0, pos);
}

std::wstring modulePath()
{
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD len = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (len == 0)
            return L"";
        if (len < path.size() - 1) {
            path.resize(len);
            return path;
        }
        path.resize(path.size() * 2);
    }
}

bool fileExists(const std::wstring& path)
{
    const DWORD attrs = GetFileAttributesW(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

std::wstring quoteArg(const std::wstring& arg)
{
    if (arg.empty())
        return L"\"\"";

    bool needsQuote = false;
    for (const wchar_t ch : arg) {
        if (ch == L' ' || ch == L'\t' || ch == L'"') {
            needsQuote = true;
            break;
        }
    }
    if (!needsQuote)
        return arg;

    std::wstring out = L"\"";
    size_t backslashes = 0;
    for (const wchar_t ch : arg) {
        if (ch == L'\\') {
            ++backslashes;
        } else if (ch == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(ch);
            backslashes = 0;
        } else {
            out.append(backslashes, L'\\');
            backslashes = 0;
            out.push_back(ch);
        }
    }
    out.append(backslashes * 2, L'\\');
    out.push_back(L'"');
    return out;
}

std::wstring forwardedArguments()
{
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv)
        return L"";

    std::wstring args;
    for (int i = 1; i < argc; ++i) {
        if (!args.empty())
            args.push_back(L' ');
        args += quoteArg(argv[i]);
    }
    LocalFree(argv);
    return args;
}

std::string narrow(const std::wstring& text)
{
    if (text.empty())
        return {};

    const int len = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1)
        return {};

    std::string out(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, out.data(), len, nullptr, nullptr);
    return out;
}

std::string lastErrorText(DWORD err)
{
    LPWSTR buffer = nullptr;
    const DWORD len = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPWSTR>(&buffer),
        0, nullptr);

    std::wstring message = len > 0 && buffer ? std::wstring(buffer, len) : L"unknown error";
    if (buffer)
        LocalFree(buffer);
    while (!message.empty()
        && (message.back() == L'\r' || message.back() == L'\n' || message.back() == L' '))
        message.pop_back();
    return narrow(message);
}

void appendLog(const std::wstring& appDir, const std::string& line)
{
    std::ofstream log(narrow(appDir + L"\\ocr_test_runtime.log"), std::ios::app);
    if (!log)
        return;

    SYSTEMTIME now{};
    GetLocalTime(&now);
    log << now.wYear << '-' << (now.wMonth < 10 ? "0" : "") << now.wMonth << '-'
        << (now.wDay < 10 ? "0" : "") << now.wDay << ' ' << (now.wHour < 10 ? "0" : "") << now.wHour
        << ':' << (now.wMinute < 10 ? "0" : "") << now.wMinute << ':'
        << (now.wSecond < 10 ? "0" : "") << now.wSecond << "  " << line << '\n';
}

bool openclRuntimeLoads(const std::wstring& appDir, std::string& reason)
{
    const std::wstring afOpenCl = appDir + L"\\afopencl.dll";
    if (!fileExists(afOpenCl)) {
        reason = "afopencl.dll not found beside ocr_test.exe";
        return false;
    }

    SetDllDirectoryW(appDir.c_str());
    HMODULE module = LoadLibraryW(afOpenCl.c_str());
    if (!module) {
        const DWORD err = GetLastError();
        reason = "cannot load afopencl.dll: " + lastErrorText(err);
        return false;
    }

    FreeLibrary(module);
    reason = "ArrayFire/OpenCL runtime detected";
    return true;
}

bool launchProcess(const std::wstring& exePath, const std::wstring& appDir, std::string& reason)
{
    if (!fileExists(exePath)) {
        reason = "executable not found: " + narrow(exePath);
        return false;
    }

    std::wstring commandLine = quoteArg(exePath);
    const std::wstring args = forwardedArguments();
    if (!args.empty()) {
        commandLine.push_back(L' ');
        commandLine += args;
    }

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    std::vector<wchar_t> mutableCommand(commandLine.begin(), commandLine.end());
    mutableCommand.push_back(L'\0');

    if (!CreateProcessW(exePath.c_str(), mutableCommand.data(), nullptr, nullptr, FALSE, 0, nullptr,
            appDir.c_str(), &si, &pi)) {
        const DWORD err = GetLastError();
        reason = "cannot start " + narrow(exePath) + ": " + lastErrorText(err);
        return false;
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    reason = "started " + narrow(exePath);
    return true;
}

void showFatal(const std::wstring& message)
{
    MessageBoxW(nullptr, message.c_str(), L"nunn OCR runtime", MB_ICONERROR | MB_OK);
}

} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, LPWSTR, int)
{
    const std::wstring launcher = modulePath();
    const std::wstring appDir = parentDir(launcher);
    const std::wstring openclExe = appDir + L"\\ocr_test_opencl.exe";
    const std::wstring cpuExe = appDir + L"\\ocr_test_cpu.exe";

    std::string reason;
    if (openclRuntimeLoads(appDir, reason)) {
        appendLog(appDir, "OpenCL candidate: " + reason);
        if (launchProcess(openclExe, appDir, reason)) {
            appendLog(appDir, reason);
            return 0;
        }
        appendLog(appDir, reason);
    } else {
        appendLog(appDir, "OpenCL disabled: " + reason);
    }

    if (launchProcess(cpuExe, appDir, reason)) {
        appendLog(appDir, "CPU fallback: " + reason);
        return 0;
    }

    appendLog(appDir, "CPU fallback failed: " + reason);
    showFatal(L"Cannot start nunn OCR.\n\n"
              L"OpenCL runtime is unavailable and ocr_test_cpu.exe could not be started.\n"
              L"See ocr_test_runtime.log in the application folder for details.");
    return 1;
}
