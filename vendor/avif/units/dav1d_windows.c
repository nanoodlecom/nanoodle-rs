#ifdef _WIN32
#include "../dav1d/src/win32/thread.c"
#else
typedef int nanoodle_no_win32_threads;
#endif
