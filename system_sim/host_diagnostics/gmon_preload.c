/* Host-only statistical profiling of a NEW diagnostic process.
 * Do not preload into an acceptance run or use samples as target cycles.
 * No simulator source, event, memory reply, or architectural state is changed.
 * glibc samples executable PCs; shared-library time and call arcs are absent.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <link.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/gmon.h>
#include <sys/time.h>
#include <unistd.h>

static int enabled;

static void fail(const char *message) {
  fprintf(stderr, "MLX_HOST_PROFILE_ERROR %s\n", message);
  _exit(125);
}

static int executable(struct dl_phdr_info *info, size_t size, void *unused) {
  (void)size;
  (void)unused;
  if (info->dlpi_name && info->dlpi_name[0]) return 0;
  uintptr_t low = UINTPTR_MAX, high = 0;
  for (unsigned i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) *header = &info->dlpi_phdr[i];
    if (header->p_type != PT_LOAD || !(header->p_flags & PF_X)) continue;
    uintptr_t begin = info->dlpi_addr + header->p_vaddr;
    uintptr_t end = begin + header->p_memsz;
    if (end < begin) fail("executable address overflow");
    if (begin < low) low = begin;
    if (end > high) high = end;
  }
  if (low >= high || high - low > 256UL * 1024 * 1024)
    fail("invalid or excessive executable sampling range");
  const char *path = getenv("MLX_HOST_PROFILE_META");
  const char *prefix = getenv("GMON_OUT_PREFIX");
  if (!path || path[0] != '/' || !prefix || prefix[0] != '/')
    fail("explicit absolute metadata and gmon paths required");
  int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) fail("metadata must be a fresh writable file");
  FILE *output = fdopen(descriptor, "w");
  if (!output) fail("metadata stream failed");
  fprintf(output, "{\"classification\":\"host_pc_samples_not_target_performance\","
          "\"pid\":%ld,\"load_bias\":%lu,\"low_pc\":%lu,\"high_pc\":%lu,"
          "\"shared_library_samples_included\":false,\"call_arcs_included\":false}\n",
          (long)getpid(), (unsigned long)info->dlpi_addr,
          (unsigned long)low, (unsigned long)high);
  if (fclose(output)) fail("metadata write failed");
  __monstartup(low, high);
  enabled = 1;
  return 1;
}

__attribute__((constructor)) static void start_sampling(void) {
  struct itimerval timer;
  struct sigaction action;
  if (getitimer(ITIMER_PROF, &timer) || sigaction(SIGPROF, NULL, &action))
    fail("cannot inspect profiler ownership");
  if (timer.it_value.tv_sec || timer.it_value.tv_usec ||
      timer.it_interval.tv_sec || timer.it_interval.tv_usec ||
      action.sa_handler != SIG_DFL)
    fail("process already owns SIGPROF or ITIMER_PROF");
  dl_iterate_phdr(executable, NULL);
  if (!enabled) fail("executable mapping not found");
}

__attribute__((destructor)) static void stop_sampling(void) {
  if (enabled) {
    _mcleanup();
  }
}
