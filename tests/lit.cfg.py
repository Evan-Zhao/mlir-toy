# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

config.name = "NEPTUNE_MLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.neptune_mlir_obj_root, "tests")

config.excludes = [
    "Inputs",
    "__pycache__",
    "data",
    "golden",
    "README.md",
]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(
    (
        "%neptune_linalg_ext_plugin",
        os.path.join(
            config.neptune_mlir_obj_root,
            "libLinalgExtTransform" + config.neptune_mlir_shared_library_suffix,
        ),
    )
)

llvm_config.add_tool_substitutions(
    ["mlir-opt", "FileCheck", "not"],
    [config.llvm_tools_dir],
)
