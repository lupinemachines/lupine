#include <cuda.h>

#include "cuda_client_graphics.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace {

using GLint = int;
using GLsizei = int;
using GLboolean = unsigned char;
using GLubyte = unsigned char;
using GLintptr = std::ptrdiff_t;
using GLsizeiptr = std::ptrdiff_t;

constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
constexpr GLenum GL_TEXTURE_3D = 0x806F;
constexpr GLenum GL_TEXTURE_RECTANGLE = 0x84F5;
constexpr GLenum GL_TEXTURE_CUBE_MAP = 0x8513;
constexpr GLenum GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515;
constexpr GLenum GL_TEXTURE_2D_ARRAY = 0x8C1A;
constexpr GLenum GL_RENDERBUFFER = 0x8D41;
constexpr GLenum GL_TEXTURE_BINDING_2D = 0x8069;
constexpr GLenum GL_TEXTURE_BINDING_3D = 0x806A;
constexpr GLenum GL_TEXTURE_BINDING_RECTANGLE = 0x84F6;
constexpr GLenum GL_TEXTURE_BINDING_CUBE_MAP = 0x8514;
constexpr GLenum GL_TEXTURE_BINDING_2D_ARRAY = 0x8C1D;
constexpr GLenum GL_TEXTURE_WIDTH = 0x1000;
constexpr GLenum GL_TEXTURE_HEIGHT = 0x1001;
constexpr GLenum GL_TEXTURE_DEPTH = 0x8071;
constexpr GLenum GL_TEXTURE_INTERNAL_FORMAT = 0x1003;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_ARRAY_BUFFER_BINDING = 0x8894;
constexpr GLenum GL_BUFFER_SIZE = 0x8764;
constexpr GLenum GL_PIXEL_PACK_BUFFER = 0x88EB;
constexpr GLenum GL_PIXEL_UNPACK_BUFFER = 0x88EC;
constexpr GLenum GL_PIXEL_PACK_BUFFER_BINDING = 0x88ED;
constexpr GLenum GL_PIXEL_UNPACK_BUFFER_BINDING = 0x88EF;

constexpr GLenum GL_RED = 0x1903;
constexpr GLenum GL_RG = 0x8227;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_RED_INTEGER = 0x8D94;
constexpr GLenum GL_RG_INTEGER = 0x8228;
constexpr GLenum GL_RGBA_INTEGER = 0x8D99;
constexpr GLenum GL_BYTE = 0x1400;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_SHORT = 0x1402;
constexpr GLenum GL_UNSIGNED_SHORT = 0x1403;
constexpr GLenum GL_INT = 0x1404;
constexpr GLenum GL_UNSIGNED_INT = 0x1405;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_HALF_FLOAT = 0x140B;

// Internal formats.
constexpr GLint GL_R8 = 0x8229;
constexpr GLint GL_RG8 = 0x822B;
constexpr GLint GL_RGB8 = 0x8051;
constexpr GLint GL_RGBA8 = 0x8058;
constexpr GLint GL_SRGB8_ALPHA8 = 0x8C43;
constexpr GLint GL_RGB = 0x1907;
constexpr GLint GL_R8I = 0x8231;
constexpr GLint GL_R8UI = 0x8232;
constexpr GLint GL_RG8I = 0x8237;
constexpr GLint GL_RG8UI = 0x8238;
constexpr GLint GL_RGBA8I = 0x8D8E;
constexpr GLint GL_RGBA8UI = 0x8D7C;
constexpr GLint GL_R16 = 0x822A;
constexpr GLint GL_RG16 = 0x822C;
constexpr GLint GL_RGBA16 = 0x805B;
constexpr GLint GL_R16I = 0x8233;
constexpr GLint GL_R16UI = 0x8234;
constexpr GLint GL_RG16I = 0x8239;
constexpr GLint GL_RG16UI = 0x823A;
constexpr GLint GL_RGBA16I = 0x8D88;
constexpr GLint GL_RGBA16UI = 0x8D76;
constexpr GLint GL_R16F = 0x822D;
constexpr GLint GL_RG16F = 0x822F;
constexpr GLint GL_RGB16F = 0x881B;
constexpr GLint GL_RGBA16F = 0x881A;
constexpr GLint GL_R32F = 0x822E;
constexpr GLint GL_RG32F = 0x8230;
constexpr GLint GL_RGB32F = 0x8815;
constexpr GLint GL_RGBA32F = 0x8814;
constexpr GLint GL_R32I = 0x8235;
constexpr GLint GL_R32UI = 0x8236;
constexpr GLint GL_RG32I = 0x823B;
constexpr GLint GL_RG32UI = 0x823C;
constexpr GLint GL_RGBA32I = 0x8D82;
constexpr GLint GL_RGBA32UI = 0x8D70;

// Pixel-store state a transfer resets to tight packing, as {pack, unpack}.
constexpr GLenum kPixelStore[][2] = {
    {0x0D05, 0x0CF5}, // ALIGNMENT
    {0x0D02, 0x0CF2}, // ROW_LENGTH
    {0x0D03, 0x0CF3}, // SKIP_ROWS
    {0x0D04, 0x0CF4}, // SKIP_PIXELS
    {0x806C, 0x806E}, // IMAGE_HEIGHT
    {0x806B, 0x806D}, // SKIP_IMAGES
};

struct gl_api {
  const GLubyte *(*GetString)(GLenum);
  GLenum (*GetError)();
  void (*GetIntegerv)(GLenum, GLint *);
  GLboolean (*IsTexture)(GLuint);
  GLboolean (*IsBuffer)(GLuint);
  void (*BindTexture)(GLenum, GLuint);
  void (*BindBuffer)(GLenum, GLuint);
  void (*PixelStorei)(GLenum, GLint);
  void (*GetTexLevelParameteriv)(GLenum, GLint, GLenum, GLint *);
  void (*GetTexImage)(GLenum, GLint, GLenum, GLenum, void *);
  void (*TexSubImage2D)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum,
                        GLenum, const void *);
  void (*TexSubImage3D)(GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei,
                        GLsizei, GLenum, GLenum, const void *);
  void (*GetBufferParameteriv)(GLenum, GLenum, GLint *);
  void (*GetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void *);
  void (*BufferSubData)(GLenum, GLintptr, GLsizeiptr, const void *);
};

#ifdef __linux__
void *gl_symbol(const char *name) {
  // libglvnd's libOpenGL and libGL both dispatch to whichever context is
  // current, whether EGL or GLX made it current.
  static void *const libraries[] = {
      dlopen("libOpenGL.so.0", RTLD_LAZY | RTLD_LOCAL),
      dlopen("libGL.so.1", RTLD_LAZY | RTLD_LOCAL),
  };
  using get_proc_address = void *(*)(const char *);
  static void *const egl = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_LOCAL);
  static const auto egl_get_proc = reinterpret_cast<get_proc_address>(
      egl != nullptr ? dlsym(egl, "eglGetProcAddress") : nullptr);
  static const auto glx_get_proc = reinterpret_cast<get_proc_address>(
      libraries[1] != nullptr ? dlsym(libraries[1], "glXGetProcAddressARB")
                              : nullptr);
  for (void *library : libraries) {
    if (library == nullptr)
      continue;
    if (void *symbol = dlsym(library, name))
      return symbol;
  }
  for (get_proc_address get_proc : {egl_get_proc, glx_get_proc}) {
    if (get_proc == nullptr)
      continue;
    if (void *symbol = get_proc(name))
      return symbol;
  }
  return nullptr;
}
#else
// Only Linux clients stage OpenGL objects.
void *gl_symbol(const char *) { return nullptr; }
#endif

template <typename T> bool load(T &slot, const char *name) {
  slot = reinterpret_cast<T>(gl_symbol(name));
  return slot != nullptr;
}

const gl_api *load_gl() {
  static const std::unique_ptr<gl_api> api = [] {
    auto gl = std::make_unique<gl_api>();
    bool ok = load(gl->GetString, "glGetString") &&
              load(gl->GetError, "glGetError") &&
              load(gl->GetIntegerv, "glGetIntegerv") &&
              load(gl->IsTexture, "glIsTexture") &&
              load(gl->IsBuffer, "glIsBuffer") &&
              load(gl->BindTexture, "glBindTexture") &&
              load(gl->BindBuffer, "glBindBuffer") &&
              load(gl->PixelStorei, "glPixelStorei") &&
              load(gl->GetTexLevelParameteriv, "glGetTexLevelParameteriv") &&
              load(gl->GetTexImage, "glGetTexImage") &&
              load(gl->TexSubImage2D, "glTexSubImage2D") &&
              load(gl->TexSubImage3D, "glTexSubImage3D") &&
              load(gl->GetBufferParameteriv, "glGetBufferParameteriv") &&
              load(gl->GetBufferSubData, "glGetBufferSubData") &&
              load(gl->BufferSubData, "glBufferSubData");
    return ok ? std::move(gl) : nullptr;
  }();
  return api.get();
}

// The GL API when the calling thread has a current context, as the driver
// requires, or null.
const gl_api *current_gl() {
  const gl_api *gl = load_gl();
  // glGetString answers null without a current context.
  return gl != nullptr && gl->GetString(GL_VERSION) != nullptr ? gl : nullptr;
}

void clear_gl_errors(const gl_api &gl) {
  for (int i = 0; i < 32 && gl.GetError() != GL_NO_ERROR; ++i) {
  }
}

GLenum texture_binding(GLenum target) {
  switch (target) {
  case GL_TEXTURE_2D:
    return GL_TEXTURE_BINDING_2D;
  case GL_TEXTURE_3D:
    return GL_TEXTURE_BINDING_3D;
  case GL_TEXTURE_RECTANGLE:
    return GL_TEXTURE_BINDING_RECTANGLE;
  case GL_TEXTURE_CUBE_MAP:
    return GL_TEXTURE_BINDING_CUBE_MAP;
  case GL_TEXTURE_2D_ARRAY:
    return GL_TEXTURE_BINDING_2D_ARRAY;
  }
  return 0;
}

// Binds one texture or buffer and resets pixel-store state for tightly packed
// client-memory transfers, restoring the application's state on destruction.
class gl_scope {
public:
  gl_scope(const gl_api &gl, GLenum target, GLuint name, bool buffer)
      : gl_(gl), target_(target), buffer_(buffer) {
    clear_gl_errors(gl_);
    gl_.GetIntegerv(buffer_ ? GL_ARRAY_BUFFER_BINDING
                            : texture_binding(target_),
                    &previous_);
    gl_.GetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack_buffer_);
    gl_.GetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &unpack_buffer_);
    gl_.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    gl_.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    for (size_t i = 0; i < std::size(kPixelStore); ++i) {
      for (int side = 0; side < 2; ++side) {
        gl_.GetIntegerv(kPixelStore[i][side], &pixel_store_[i][side]);
        gl_.PixelStorei(kPixelStore[i][side], i == 0 ? 1 : 0);
      }
    }
    if (buffer_)
      gl_.BindBuffer(GL_ARRAY_BUFFER, name);
    else
      gl_.BindTexture(target_, name);
  }

  ~gl_scope() {
    if (buffer_)
      gl_.BindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(previous_));
    else
      gl_.BindTexture(target_, static_cast<GLuint>(previous_));
    for (size_t i = 0; i < std::size(kPixelStore); ++i) {
      for (int side = 0; side < 2; ++side)
        gl_.PixelStorei(kPixelStore[i][side], pixel_store_[i][side]);
    }
    gl_.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(pack_buffer_));
    gl_.BindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(unpack_buffer_));
  }

  // Whether every GL call since the scope opened succeeded.
  bool ok() const { return gl_.GetError() == GL_NO_ERROR; }

private:
  const gl_api &gl_;
  GLenum target_;
  bool buffer_;
  GLint previous_ = 0;
  GLint pack_buffer_ = 0;
  GLint unpack_buffer_ = 0;
  GLint pixel_store_[std::size(kPixelStore)][2] = {};
};

// How a GL internal format is held in a CUDA array and moved between them.
// Three-channel formats are padded to four, as the driver does.
struct texture_format {
  GLint internal_format;
  CUarray_format array_format;
  unsigned int channels;
  GLenum transfer_format;
  GLenum transfer_type;
};

constexpr texture_format kFormats[] = {
    {GL_R8, CU_AD_FORMAT_UNSIGNED_INT8, 1, GL_RED, GL_UNSIGNED_BYTE},
    {GL_RG8, CU_AD_FORMAT_UNSIGNED_INT8, 2, GL_RG, GL_UNSIGNED_BYTE},
    {GL_RGB8, CU_AD_FORMAT_UNSIGNED_INT8, 4, GL_RGBA, GL_UNSIGNED_BYTE},
    {GL_RGBA8, CU_AD_FORMAT_UNSIGNED_INT8, 4, GL_RGBA, GL_UNSIGNED_BYTE},
    {GL_SRGB8_ALPHA8, CU_AD_FORMAT_UNSIGNED_INT8, 4, GL_RGBA, GL_UNSIGNED_BYTE},
    {GL_RGB, CU_AD_FORMAT_UNSIGNED_INT8, 4, GL_RGBA, GL_UNSIGNED_BYTE},
    {GL_RGBA, CU_AD_FORMAT_UNSIGNED_INT8, 4, GL_RGBA, GL_UNSIGNED_BYTE},
    {GL_R8I, CU_AD_FORMAT_SIGNED_INT8, 1, GL_RED_INTEGER, GL_BYTE},
    {GL_R8UI, CU_AD_FORMAT_UNSIGNED_INT8, 1, GL_RED_INTEGER, GL_UNSIGNED_BYTE},
    {GL_RG8I, CU_AD_FORMAT_SIGNED_INT8, 2, GL_RG_INTEGER, GL_BYTE},
    {GL_RG8UI, CU_AD_FORMAT_UNSIGNED_INT8, 2, GL_RG_INTEGER, GL_UNSIGNED_BYTE},
    {GL_RGBA8I, CU_AD_FORMAT_SIGNED_INT8, 4, GL_RGBA_INTEGER, GL_BYTE},
    {GL_RGBA8UI, CU_AD_FORMAT_UNSIGNED_INT8, 4, GL_RGBA_INTEGER,
     GL_UNSIGNED_BYTE},
    {GL_R16, CU_AD_FORMAT_UNSIGNED_INT16, 1, GL_RED, GL_UNSIGNED_SHORT},
    {GL_RG16, CU_AD_FORMAT_UNSIGNED_INT16, 2, GL_RG, GL_UNSIGNED_SHORT},
    {GL_RGBA16, CU_AD_FORMAT_UNSIGNED_INT16, 4, GL_RGBA, GL_UNSIGNED_SHORT},
    {GL_R16I, CU_AD_FORMAT_SIGNED_INT16, 1, GL_RED_INTEGER, GL_SHORT},
    {GL_R16UI, CU_AD_FORMAT_UNSIGNED_INT16, 1, GL_RED_INTEGER,
     GL_UNSIGNED_SHORT},
    {GL_RG16I, CU_AD_FORMAT_SIGNED_INT16, 2, GL_RG_INTEGER, GL_SHORT},
    {GL_RG16UI, CU_AD_FORMAT_UNSIGNED_INT16, 2, GL_RG_INTEGER,
     GL_UNSIGNED_SHORT},
    {GL_RGBA16I, CU_AD_FORMAT_SIGNED_INT16, 4, GL_RGBA_INTEGER, GL_SHORT},
    {GL_RGBA16UI, CU_AD_FORMAT_UNSIGNED_INT16, 4, GL_RGBA_INTEGER,
     GL_UNSIGNED_SHORT},
    {GL_R16F, CU_AD_FORMAT_HALF, 1, GL_RED, GL_HALF_FLOAT},
    {GL_RG16F, CU_AD_FORMAT_HALF, 2, GL_RG, GL_HALF_FLOAT},
    {GL_RGB16F, CU_AD_FORMAT_HALF, 4, GL_RGBA, GL_HALF_FLOAT},
    {GL_RGBA16F, CU_AD_FORMAT_HALF, 4, GL_RGBA, GL_HALF_FLOAT},
    {GL_R32F, CU_AD_FORMAT_FLOAT, 1, GL_RED, GL_FLOAT},
    {GL_RG32F, CU_AD_FORMAT_FLOAT, 2, GL_RG, GL_FLOAT},
    {GL_RGB32F, CU_AD_FORMAT_FLOAT, 4, GL_RGBA, GL_FLOAT},
    {GL_RGBA32F, CU_AD_FORMAT_FLOAT, 4, GL_RGBA, GL_FLOAT},
    {GL_R32I, CU_AD_FORMAT_SIGNED_INT32, 1, GL_RED_INTEGER, GL_INT},
    {GL_R32UI, CU_AD_FORMAT_UNSIGNED_INT32, 1, GL_RED_INTEGER, GL_UNSIGNED_INT},
    {GL_RG32I, CU_AD_FORMAT_SIGNED_INT32, 2, GL_RG_INTEGER, GL_INT},
    {GL_RG32UI, CU_AD_FORMAT_UNSIGNED_INT32, 2, GL_RG_INTEGER, GL_UNSIGNED_INT},
    {GL_RGBA32I, CU_AD_FORMAT_SIGNED_INT32, 4, GL_RGBA_INTEGER, GL_INT},
    {GL_RGBA32UI, CU_AD_FORMAT_UNSIGNED_INT32, 4, GL_RGBA_INTEGER,
     GL_UNSIGNED_INT},
};

size_t element_bytes(CUarray_format format) {
  switch (format) {
  case CU_AD_FORMAT_UNSIGNED_INT8:
  case CU_AD_FORMAT_SIGNED_INT8:
    return 1;
  case CU_AD_FORMAT_UNSIGNED_INT16:
  case CU_AD_FORMAT_SIGNED_INT16:
  case CU_AD_FORMAT_HALF:
    return 2;
  default:
    return 4;
  }
}

struct level_extent {
  size_t width;
  size_t height;
  // Slices in one GL transfer: the depth of a 3D texture, the layers of an
  // array texture, the six faces of a cube map, and 1 otherwise.
  size_t slices;
  // Whether GL has storage at this level. The driver maps the whole mip
  // chain, but only levels GL defines have contents to move.
  bool defined;
};

struct staged_resource {
  bool image = false;
  GLuint name = 0;
  GLenum target = 0;
  unsigned int map_flags = 0;
  bool mapped = false;

  // Images keep one mipmapped array per layer or cube face, so each
  // subresource the driver would hand out is a level of one of them.
  const texture_format *format = nullptr;
  std::vector<level_extent> levels;
  std::vector<CUmipmappedArray> layers;

  // Buffers keep one device allocation.
  CUdeviceptr device = 0;
  size_t size = 0;

  size_t pixel_bytes() const {
    return element_bytes(format->array_format) * format->channels;
  }
  size_t slice_bytes(const level_extent &level) const {
    return level.width * level.height * pixel_bytes();
  }
};

std::mutex staged_mutex;
std::unordered_map<CUgraphicsResource, std::unique_ptr<staged_resource>>
    staged_resources;

staged_resource *find_staged(CUgraphicsResource resource) {
  auto it = staged_resources.find(resource);
  return it != staged_resources.end() ? it->second.get() : nullptr;
}

CUgraphicsResource add_staged(std::unique_ptr<staged_resource> staged) {
  auto handle = reinterpret_cast<CUgraphicsResource>(staged.get());
  staged_resources.emplace(handle, std::move(staged));
  return handle;
}

void release_storage(staged_resource &staged) {
  for (CUmipmappedArray layer : staged.layers)
    cuMipmappedArrayDestroy(layer);
  staged.layers.clear();
  if (staged.device != 0)
    cuMemFree(staged.device);
  staged.device = 0;
}

CUresult require_cuda_context() {
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS)
    return result;
  return context != nullptr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}

// The texture's GL target for one layer: a cube map is read and written face
// by face.
GLenum layer_target(const staged_resource &staged, size_t layer) {
  return staged.target == GL_TEXTURE_CUBE_MAP
             ? static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer)
             : staged.target;
}

// The copy between one array level and its slice of a host level image.
CUDA_MEMCPY3D array_copy(const staged_resource &staged,
                         const level_extent &level, size_t layer, CUarray array,
                         unsigned char *host, bool to_array) {
  bool volume = staged.target == GL_TEXTURE_3D;
  size_t row = level.width * staged.pixel_bytes();
  CUDA_MEMCPY3D copy = {};
  void *slice = host + (volume ? 0 : layer * staged.slice_bytes(level));
  if (to_array) {
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = slice;
    copy.srcPitch = row;
    copy.srcHeight = level.height;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
  } else {
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = slice;
    copy.dstPitch = row;
    copy.dstHeight = level.height;
  }
  copy.WidthInBytes = row;
  copy.Height = level.height;
  copy.Depth = volume ? level.slices : 1;
  return copy;
}

// Enqueues the copies between the GL image and its arrays. Host images are
// appended to `hosts`, which must outlive the stream's completion.
CUresult stage_image(const gl_api &gl, staged_resource &staged, CUstream stream,
                     bool upload,
                     std::vector<std::vector<unsigned char>> &hosts) {
  gl_scope scope(gl, staged.target, staged.name, false);
  for (size_t level = 0; level < staged.levels.size(); ++level) {
    const level_extent &extent = staged.levels[level];
    hosts.emplace_back(
        extent.defined ? staged.slice_bytes(extent) * extent.slices : 0);
    if (!extent.defined)
      continue;
    unsigned char *host = hosts.back().data();
    if (upload) {
      if (staged.target == GL_TEXTURE_CUBE_MAP) {
        for (size_t face = 0; face < extent.slices; ++face)
          gl.GetTexImage(layer_target(staged, face), static_cast<GLint>(level),
                         staged.format->transfer_format,
                         staged.format->transfer_type,
                         host + face * staged.slice_bytes(extent));
      } else {
        gl.GetTexImage(staged.target, static_cast<GLint>(level),
                       staged.format->transfer_format,
                       staged.format->transfer_type, host);
      }
      if (!scope.ok())
        return CUDA_ERROR_UNKNOWN;
    }
    for (size_t layer = 0; layer < staged.layers.size(); ++layer) {
      CUarray array = nullptr;
      CUresult result = cuMipmappedArrayGetLevel(
          &array, staged.layers[layer], static_cast<unsigned int>(level));
      if (result != CUDA_SUCCESS)
        return result;
      CUDA_MEMCPY3D copy =
          array_copy(staged, extent, layer, array, host, upload);
      result = cuMemcpy3DAsync(&copy, stream);
      if (result != CUDA_SUCCESS)
        return result;
    }
  }
  return CUDA_SUCCESS;
}

// Writes host level images, which the stream has filled, back to the texture.
CUresult write_back_image(const gl_api &gl, const staged_resource &staged,
                          const std::vector<unsigned char> *hosts) {
  gl_scope scope(gl, staged.target, staged.name, false);
  for (size_t level = 0; level < staged.levels.size(); ++level) {
    const level_extent &extent = staged.levels[level];
    if (!extent.defined)
      continue;
    const unsigned char *host = hosts[level].data();
    auto width = static_cast<GLsizei>(extent.width);
    auto height = static_cast<GLsizei>(extent.height);
    auto gl_level = static_cast<GLint>(level);
    GLenum format = staged.format->transfer_format;
    GLenum type = staged.format->transfer_type;
    switch (staged.target) {
    case GL_TEXTURE_CUBE_MAP:
      for (size_t face = 0; face < extent.slices; ++face)
        gl.TexSubImage2D(layer_target(staged, face), gl_level, 0, 0, width,
                         height, format, type,
                         host + face * staged.slice_bytes(extent));
      break;
    case GL_TEXTURE_3D:
    case GL_TEXTURE_2D_ARRAY:
      gl.TexSubImage3D(staged.target, gl_level, 0, 0, 0, width, height,
                       static_cast<GLsizei>(extent.slices), format, type, host);
      break;
    default:
      gl.TexSubImage2D(staged.target, gl_level, 0, 0, width, height, format,
                       type, host);
      break;
    }
  }
  return scope.ok() ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

GLint buffer_size(const gl_api &gl, GLuint buffer) {
  gl_scope scope(gl, GL_ARRAY_BUFFER, buffer, true);
  GLint size = 0;
  gl.GetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
  return scope.ok() ? size : -1;
}

CUresult validate_resources(unsigned int count, CUgraphicsResource *resources,
                            bool mapped, std::vector<staged_resource *> &out) {
  if (count != 0 && resources == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  for (unsigned int i = 0; i < count; ++i) {
    staged_resource *staged = find_staged(resources[i]);
    if (staged == nullptr)
      return CUDA_ERROR_INVALID_HANDLE;
    if (staged->mapped != mapped)
      return mapped ? CUDA_ERROR_NOT_MAPPED : CUDA_ERROR_ALREADY_MAPPED;
    out.push_back(staged);
  }
  return CUDA_SUCCESS;
}

constexpr unsigned int kRegisterFlags =
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY |
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD |
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST |
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER;
constexpr unsigned int kMapFlags = CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY |
                                   CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD;

} // namespace

CUresult
lupine_staged_cuGraphicsGLRegisterBuffer(CUgraphicsResource *pCudaResource,
                                         GLuint buffer, unsigned int Flags) {
  if (pCudaResource == nullptr || (Flags & ~kRegisterFlags) != 0)
    return CUDA_ERROR_INVALID_VALUE;
  const gl_api *gl = current_gl();
  if (gl == nullptr)
    return CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;
  CUresult result = require_cuda_context();
  if (result != CUDA_SUCCESS)
    return result;
  if (!gl->IsBuffer(buffer))
    return CUDA_ERROR_INVALID_HANDLE;
  GLint size = buffer_size(*gl, buffer);
  if (size < 0)
    return CUDA_ERROR_INVALID_VALUE;
  // The driver reports a buffer without storage as out of memory.
  if (size == 0)
    return CUDA_ERROR_OUT_OF_MEMORY;

  auto staged = std::make_unique<staged_resource>();
  staged->name = buffer;
  staged->map_flags = Flags & kMapFlags;
  staged->size = static_cast<size_t>(size);
  result = cuMemAlloc(&staged->device, staged->size);
  if (result != CUDA_SUCCESS)
    return result;
  std::lock_guard<std::mutex> lock(staged_mutex);
  *pCudaResource = add_staged(std::move(staged));
  return CUDA_SUCCESS;
}

CUresult
lupine_staged_cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource,
                                        GLuint image, GLenum target,
                                        unsigned int Flags) {
  if (pCudaResource == nullptr || (Flags & ~kRegisterFlags) != 0)
    return CUDA_ERROR_INVALID_VALUE;
  const gl_api *gl = current_gl();
  if (gl == nullptr)
    return CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;
  CUresult result = require_cuda_context();
  if (result != CUDA_SUCCESS)
    return result;
  if (target == GL_RENDERBUFFER)
    return CUDA_ERROR_NOT_SUPPORTED;
  if (texture_binding(target) == 0)
    return CUDA_ERROR_INVALID_VALUE;
  if (!gl->IsTexture(image))
    return CUDA_ERROR_INVALID_HANDLE;

  auto staged = std::make_unique<staged_resource>();
  staged->image = true;
  staged->name = image;
  staged->target = target;
  staged->map_flags = Flags & kMapFlags;
  GLint internal_format = 0;
  {
    gl_scope scope(*gl, target, image, false);
    GLenum query = layer_target(*staged, 0);
    gl->GetTexLevelParameteriv(query, 0, GL_TEXTURE_INTERNAL_FORMAT,
                               &internal_format);
    GLint base[3] = {};
    gl->GetTexLevelParameteriv(query, 0, GL_TEXTURE_WIDTH, &base[0]);
    gl->GetTexLevelParameteriv(query, 0, GL_TEXTURE_HEIGHT, &base[1]);
    gl->GetTexLevelParameteriv(query, 0, GL_TEXTURE_DEPTH, &base[2]);
    if (!scope.ok() || base[0] <= 0)
      return CUDA_ERROR_INVALID_VALUE;
    // The full mip chain of the base level, which a rectangle texture lacks.
    bool volume = target == GL_TEXTURE_3D;
    size_t largest = std::max<size_t>(base[0], base[1]);
    if (volume)
      largest = std::max<size_t>(largest, base[2]);
    size_t count = 1;
    while (target != GL_TEXTURE_RECTANGLE && (largest >> count) != 0)
      ++count;
    for (size_t level = 0; level < count; ++level) {
      level_extent extent = {};
      extent.width = std::max<size_t>(1, static_cast<size_t>(base[0]) >> level);
      extent.height =
          std::max<size_t>(1, static_cast<size_t>(base[1]) >> level);
      extent.slices =
          target == GL_TEXTURE_CUBE_MAP ? 6
          : volume ? std::max<size_t>(1, static_cast<size_t>(base[2]) >> level)
          : target == GL_TEXTURE_2D_ARRAY ? static_cast<size_t>(base[2])
                                          : 1;
      GLint width = 0, height = 0, depth = 0;
      auto gl_level = static_cast<GLint>(level);
      gl->GetTexLevelParameteriv(query, gl_level, GL_TEXTURE_WIDTH, &width);
      gl->GetTexLevelParameteriv(query, gl_level, GL_TEXTURE_HEIGHT, &height);
      gl->GetTexLevelParameteriv(query, gl_level, GL_TEXTURE_DEPTH, &depth);
      extent.defined = static_cast<size_t>(width) == extent.width &&
                       static_cast<size_t>(height) == extent.height &&
                       (!volume || static_cast<size_t>(depth) == extent.slices);
      staged->levels.push_back(extent);
    }
    if (!scope.ok())
      return CUDA_ERROR_INVALID_VALUE;
  }
  for (const texture_format &format : kFormats) {
    if (format.internal_format == internal_format)
      staged->format = &format;
  }
  if (staged->format == nullptr)
    return CUDA_ERROR_NOT_SUPPORTED;

  const level_extent &base = staged->levels[0];
  bool volume = target == GL_TEXTURE_3D;
  CUDA_ARRAY3D_DESCRIPTOR descriptor = {};
  descriptor.Width = base.width;
  descriptor.Height = base.height;
  descriptor.Depth = volume ? base.slices : 0;
  descriptor.Format = staged->format->array_format;
  descriptor.NumChannels = staged->format->channels;
  if (Flags & CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST)
    descriptor.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
  if (Flags & CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER)
    descriptor.Flags |= CUDA_ARRAY3D_TEXTURE_GATHER;
  size_t layers = volume ? 1 : base.slices;
  for (size_t layer = 0; layer < layers; ++layer) {
    CUmipmappedArray array = nullptr;
    result = cuMipmappedArrayCreate(
        &array, &descriptor, static_cast<unsigned int>(staged->levels.size()));
    if (result != CUDA_SUCCESS) {
      release_storage(*staged);
      return result;
    }
    staged->layers.push_back(array);
  }
  std::lock_guard<std::mutex> lock(staged_mutex);
  *pCudaResource = add_staged(std::move(staged));
  return CUDA_SUCCESS;
}

CUresult lupine_staged_cuGLGetDevices_v2(unsigned int *pCudaDeviceCount,
                                         CUdevice *pCudaDevices,
                                         unsigned int cudaDeviceCount,
                                         CUGLDeviceList deviceList) {
  if (deviceList < CU_GL_DEVICE_LIST_ALL ||
      deviceList > CU_GL_DEVICE_LIST_NEXT_FRAME)
    return CUDA_ERROR_INVALID_VALUE;
  if (current_gl() == nullptr)
    return CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;
  // Staging serves any device, so every device can share the GL context.
  int count = 0;
  CUresult result = cuDeviceGetCount(&count);
  if (result != CUDA_SUCCESS)
    return result;
  if (pCudaDeviceCount != nullptr)
    *pCudaDeviceCount = static_cast<unsigned int>(count);
  for (unsigned int i = 0; pCudaDevices != nullptr && i < cudaDeviceCount &&
                           i < static_cast<unsigned int>(count);
       ++i) {
    result = cuDeviceGet(&pCudaDevices[i], static_cast<int>(i));
    if (result != CUDA_SUCCESS)
      return result;
  }
  return CUDA_SUCCESS;
}

CUresult
lupine_staged_cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  std::lock_guard<std::mutex> lock(staged_mutex);
  staged_resource *staged = find_staged(resource);
  if (staged == nullptr)
    return CUDA_ERROR_INVALID_HANDLE;
  if (staged->mapped)
    return CUDA_ERROR_ALREADY_MAPPED;
  release_storage(*staged);
  staged_resources.erase(resource);
  return CUDA_SUCCESS;
}

CUresult lupine_staged_cuGraphicsSubResourceGetMappedArray(
    CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  if (pArray == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(staged_mutex);
  staged_resource *staged = find_staged(resource);
  if (staged == nullptr)
    return CUDA_ERROR_INVALID_HANDLE;
  if (!staged->mapped)
    return CUDA_ERROR_NOT_MAPPED;
  if (!staged->image)
    return CUDA_ERROR_NOT_MAPPED_AS_ARRAY;
  if (arrayIndex >= staged->layers.size() || mipLevel >= staged->levels.size())
    return CUDA_ERROR_INVALID_VALUE;
  return cuMipmappedArrayGetLevel(pArray, staged->layers[arrayIndex], mipLevel);
}

CUresult lupine_staged_cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
  if (pMipmappedArray == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(staged_mutex);
  staged_resource *staged = find_staged(resource);
  if (staged == nullptr)
    return CUDA_ERROR_INVALID_HANDLE;
  if (!staged->mapped)
    return CUDA_ERROR_NOT_MAPPED;
  if (!staged->image)
    return CUDA_ERROR_NOT_MAPPED_AS_ARRAY;
  // Cube maps and array textures are staged a face or layer at a time, which
  // no single mipmapped array describes.
  if (staged->target == GL_TEXTURE_CUBE_MAP ||
      staged->target == GL_TEXTURE_2D_ARRAY)
    return CUDA_ERROR_NOT_SUPPORTED;
  *pMipmappedArray = staged->layers[0];
  return CUDA_SUCCESS;
}

CUresult lupine_staged_cuGraphicsResourceGetMappedPointer_v2(
    CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
  std::lock_guard<std::mutex> lock(staged_mutex);
  staged_resource *staged = find_staged(resource);
  if (staged == nullptr)
    return CUDA_ERROR_INVALID_HANDLE;
  if (!staged->mapped)
    return CUDA_ERROR_NOT_MAPPED;
  if (staged->image)
    return CUDA_ERROR_NOT_MAPPED_AS_POINTER;
  if (pDevPtr != nullptr)
    *pDevPtr = staged->device;
  if (pSize != nullptr)
    *pSize = staged->size;
  return CUDA_SUCCESS;
}

CUresult
lupine_staged_cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                               unsigned int flags) {
  std::lock_guard<std::mutex> lock(staged_mutex);
  staged_resource *staged = find_staged(resource);
  if (staged == nullptr)
    return CUDA_ERROR_INVALID_HANDLE;
  if (flags > CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
    return CUDA_ERROR_INVALID_VALUE;
  if (staged->mapped)
    return CUDA_ERROR_ALREADY_MAPPED;
  staged->map_flags = flags;
  return CUDA_SUCCESS;
}

CUresult lupine_staged_cuGraphicsMapResources(unsigned int count,
                                              CUgraphicsResource *resources,
                                              CUstream hStream) {
  std::lock_guard<std::mutex> lock(staged_mutex);
  std::vector<staged_resource *> staged;
  CUresult result = validate_resources(count, resources, false, staged);
  if (result != CUDA_SUCCESS)
    return result;
  const gl_api *gl = current_gl();
  if (gl == nullptr)
    return CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;

  std::vector<std::vector<unsigned char>> hosts;
  for (staged_resource *resource : staged) {
    // A write-discard mapping overwrites everything, so it needs no upload.
    if (resource->map_flags & CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
      continue;
    if (resource->image) {
      result = stage_image(*gl, *resource, hStream, true, hosts);
    } else {
      hosts.emplace_back(resource->size);
      {
        gl_scope scope(*gl, GL_ARRAY_BUFFER, resource->name, true);
        gl->GetBufferSubData(GL_ARRAY_BUFFER, 0,
                             static_cast<GLsizeiptr>(resource->size),
                             hosts.back().data());
        if (!scope.ok())
          return CUDA_ERROR_UNKNOWN;
      }
      result = cuMemcpyHtoDAsync(resource->device, hosts.back().data(),
                                 resource->size, hStream);
    }
    if (result != CUDA_SUCCESS)
      break;
  }
  CUresult synchronized = cuStreamSynchronize(hStream);
  if (result != CUDA_SUCCESS)
    return result;
  if (synchronized != CUDA_SUCCESS)
    return synchronized;
  for (staged_resource *resource : staged)
    resource->mapped = true;
  return CUDA_SUCCESS;
}

CUresult lupine_staged_cuGraphicsUnmapResources(unsigned int count,
                                                CUgraphicsResource *resources,
                                                CUstream hStream) {
  std::lock_guard<std::mutex> lock(staged_mutex);
  std::vector<staged_resource *> staged;
  CUresult result = validate_resources(count, resources, true, staged);
  if (result != CUDA_SUCCESS)
    return result;
  const gl_api *gl = current_gl();
  if (gl == nullptr)
    return CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;

  // Each written resource's host images, filled by the stream before GL reads
  // them.
  std::vector<std::vector<std::vector<unsigned char>>> hosts(staged.size());
  for (size_t i = 0; i < staged.size() && result == CUDA_SUCCESS; ++i) {
    staged_resource *resource = staged[i];
    // A read-only mapping left the GL object as it was.
    if (resource->map_flags & CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY)
      continue;
    if (resource->image) {
      result = stage_image(*gl, *resource, hStream, false, hosts[i]);
    } else {
      hosts[i].emplace_back(resource->size);
      result = cuMemcpyDtoHAsync(hosts[i].back().data(), resource->device,
                                 resource->size, hStream);
    }
  }
  CUresult synchronized = cuStreamSynchronize(hStream);
  if (result != CUDA_SUCCESS)
    return result;
  if (synchronized != CUDA_SUCCESS)
    return synchronized;

  for (size_t i = 0; i < staged.size(); ++i) {
    staged_resource *resource = staged[i];
    if (hosts[i].empty())
      continue;
    if (resource->image) {
      result = write_back_image(*gl, *resource, hosts[i].data());
    } else {
      gl_scope scope(*gl, GL_ARRAY_BUFFER, resource->name, true);
      gl->BufferSubData(GL_ARRAY_BUFFER, 0,
                        static_cast<GLsizeiptr>(resource->size),
                        hosts[i][0].data());
      result = scope.ok() ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }
    if (result != CUDA_SUCCESS)
      return result;
  }
  for (staged_resource *resource : staged)
    resource->mapped = false;
  return CUDA_SUCCESS;
}
