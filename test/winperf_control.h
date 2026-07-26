// Vendored from https://github.com/kjk/winperf client/winperf_control.h.
// Keep in sync with ..\winperf\client\winperf_control.h when that changes.
//
// winperf_control.h — drop-in, single-header control for winperf "section" profiling.
//
// Add calls around the code you want profiled:
//
//     winperf_profile_start();
//     ... work you care about ...
//     winperf_profile_stop();
//
// In C++ you can instead use a scoped guard that stops on scope exit:
//
//     { WinperfProfileRegion _r; ... work ... }
//
// When the program is NOT running under `winperf record`, every call is a cheap
// no-op (one failed pipe open, then cached off). No winperf headers or libs are
// needed — just drop this file into the target and #include it.
//
// How it works: winperf keeps one system-wide ETW session running the whole time
// and drops every sample taken outside a marked region. Each call here takes a
// QueryPerformanceCounter() timestamp AT THE CALL SITE and ships it to winperf
// over a named pipe. Because the timestamp is captured in-process, region
// boundaries are exact regardless of IPC latency — and QPC is the same clock and
// units as the raw ETL sample timestamps, so no conversion is needed.
//
// Regions may nest and may span threads; nested start/stop pairs collapse into
// the outermost region. Marks are tagged with GetCurrentProcessId(), so winperf
// filters per process.
//
// Thread-safety: safe to call from any thread. The first call races to open the
// pipe; if two threads make the very first call simultaneously one of those marks
// may be dropped. In practice the first start is issued from a single thread.

#ifndef WINPERF_CONTROL_H
#define WINPERF_CONTROL_H

#include <windows.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wire format: one fixed 13-byte record per mark. Keep in sync with WinperfMark
// in the winperf port (src/WinperfApp.h). Packed + explicit widths so a 32-bit target
// and a 64-bit winperf agree on the layout.
#pragma pack(push, 1)
typedef struct WinperfMark_ {
    uint8_t opcode; // 0 = stop, 1 = start
    uint32_t pid;
    uint64_t qpc; // QueryPerformanceCounter() at the call site
} WinperfMark_;
#pragma pack(pop)

static void winperf__send(uint8_t opcode) {
    // s_pipe: INVALID_HANDLE_VALUE = not yet tried, NULL = tried and disabled,
    // anything else = live write handle.
    static volatile HANDLE s_pipe = INVALID_HANDLE_VALUE;
    static volatile LONG s_connecting = 0;

    HANDLE h = s_pipe;
    if (h == INVALID_HANDLE_VALUE) {
        // Connect exactly once; the loser of the race skips this mark.
        if (InterlockedCompareExchange(&s_connecting, 1, 0) != 0) {
            return;
        }
        h = CreateFileW(L"\\\\.\\pipe\\winperf-control", GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0,
                        NULL);
        if (h == INVALID_HANDLE_VALUE) {
            h = NULL; // not running under winperf -> stay disabled
        }
        s_pipe = h;
    }
    if (h == NULL) {
        return; // disabled
    }

    WinperfMark_ m;
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    m.opcode = opcode;
    m.pid = (uint32_t)GetCurrentProcessId();
    m.qpc = (uint64_t)qpc.QuadPart;
    DWORD wrote = 0;
    WriteFile(h, &m, (DWORD)sizeof(m), &wrote, NULL); // one atomic message
}

// Mark the start of a region of interest.
static void winperf_profile_start(void) {
    winperf__send(1);
}
// Mark the end of a region of interest.
static void winperf_profile_stop(void) {
    winperf__send(0);
}

#ifdef __cplusplus
} // extern "C"

// RAII helper: profiles for the lifetime of the object.
struct WinperfProfileRegion {
    WinperfProfileRegion() {
        winperf_profile_start();
    }
    ~WinperfProfileRegion() {
        winperf_profile_stop();
    }
};
#endif

#endif // WINPERF_CONTROL_H
