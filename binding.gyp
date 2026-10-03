{
  "targets": [
    {
      # The AVX2 kernels live in their own target because node-gyp applies
      # compiler flags per target, not per file — and enabling AVX2 for the
      # whole addon would make it crash on any CPU that lacks it.
      "target_name": "vecsearch_avx2",
      "type": "static_library",
      "sources": ["src/distance_avx2.cpp"],
      "include_dirs": ["include"],
      # The shared headers can throw, so this target needs exceptions on too —
      # node-gyp disables them by default for every target.
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": ["-mavx2", "-mfma", "-O3"],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "OTHER_CPLUSPLUSFLAGS": ["-mavx2", "-mfma", "-O3"]
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "EnableEnhancedInstructionSet": 5,
          "ExceptionHandling": 1
        }
      }
    },
    {
      "target_name": "vecsearch",
      "sources": [
        "binding/addon.cpp",
        "src/hnsw.cpp",
        "src/storage.cpp",
        "src/distance.cpp",
        "src/distance_scalar.cpp"
      ],
      # Two ways to find napi.h, on purpose:
      #  - `.include` is node-addon-api's own answer: an absolute, quoted path,
      #    so it resolves no matter which directory the compiler runs from.
      #    (`.include_dir` is relative, and broke on Windows: MSBuild compiles
      #    from inside build/, where "node_modules/..." does not exist.)
      #  - the plain relative path is a fallback that gyp rebases itself.
      "include_dirs": [
        "include",
        "<!@(node -p \"require('node-addon-api').include\")",
        "node_modules/node-addon-api"
      ],
      "dependencies": ["vecsearch_avx2"],
      "defines": ["NAPI_CPP_EXCEPTIONS", "NAPI_VERSION=8"],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": ["-O3"],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.15"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1
        }
      }
    }
  ]
}
