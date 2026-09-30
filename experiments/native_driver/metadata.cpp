// Native libcuda also inspects PCI and driver metadata outside /dev/nvidia*.
// run_gpu.sh copies these files from the GPU host; no CUDA handles live here.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

const char *native_driver_metadata_path(const char *path) {
  const char *root = std::getenv("LUPINE_NATIVE_METADATA");
  if (!root || !path)
    return path;
  static thread_local char buffer[4096];
  if (std::strncmp(path, "/sys/bus/pci/devices", 20) == 0) {
    std::snprintf(buffer, sizeof(buffer), "%s/pci%s", root, path + 20);
    return buffer;
  }
  const char *suffix = nullptr;
  if (std::strcmp(path, "/proc/modules") == 0)
    suffix = "nvidia-modules";
  if (std::strcmp(path, "/proc/driver/nvidia/params") == 0)
    suffix = "nvidia-params";
  if (std::strcmp(path, "/sys/module/nvidia/initstate") == 0)
    suffix = "nvidia-initstate";
  if (!suffix)
    return path;
  std::snprintf(buffer, sizeof(buffer), "%s/%s", root, suffix);
  return buffer;
}

extern "C" FILE *fopen(const char *path, const char *mode) {
  static auto next = reinterpret_cast<FILE *(*)(const char *, const char *)>(
      dlsym(RTLD_NEXT, "fopen"));
  return next(native_driver_metadata_path(path), mode);
}

extern "C" int access(const char *path, int mode) {
  static auto next =
      reinterpret_cast<int (*)(const char *, int)>(dlsym(RTLD_NEXT, "access"));
  return next(native_driver_metadata_path(path), mode);
}

extern "C" DIR *opendir(const char *path) {
  static auto next =
      reinterpret_cast<DIR *(*)(const char *)>(dlsym(RTLD_NEXT, "opendir"));
  return next(native_driver_metadata_path(path));
}

template <class Stat> static bool device_stat(const char *path, Stat *info) {
  if (std::strcmp(path, "/dev/nvidiactl") != 0 &&
      std::strcmp(path, "/dev/nvidia0") != 0 &&
      std::strcmp(path, "/dev/nvidia-uvm") != 0)
    return false;
  std::memset(info, 0, sizeof(*info));
  info->st_mode = S_IFCHR | 0666;
  info->st_rdev = makedev(std::strcmp(path, "/dev/nvidia-uvm") == 0 ? 234 : 195,
                          std::strcmp(path, "/dev/nvidiactl") == 0 ? 255 : 0);
  info->st_nlink = 1;
  return true;
}

extern "C" int stat(const char *path, struct stat *info) noexcept {
  if (device_stat(path, info))
    return 0;
  static auto next = reinterpret_cast<int (*)(const char *, struct stat *)>(
      dlsym(RTLD_NEXT, "stat"));
  return next(native_driver_metadata_path(path), info);
}

extern "C" int __xstat(int version, const char *path, struct stat *info) {
  if (device_stat(path, info))
    return 0;
  static auto next =
      reinterpret_cast<int (*)(int, const char *, struct stat *)>(
          dlsym(RTLD_NEXT, "__xstat"));
  return next(version, native_driver_metadata_path(path), info);
}

extern "C" int __xstat64(int version, const char *path, struct stat64 *info) {
  if (device_stat(path, info))
    return 0;
  static auto next =
      reinterpret_cast<int (*)(int, const char *, struct stat64 *)>(
          dlsym(RTLD_NEXT, "__xstat64"));
  return next(version, native_driver_metadata_path(path), info);
}
