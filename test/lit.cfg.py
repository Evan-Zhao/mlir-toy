# -*- Python -*-

import os
import sysconfig

import lit.formats
from lit.llvm import llvm_config

config.name = "NEPTUNE_MLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.neptune_mlir_obj_root, "tests")

config.excludes = ["python", "Inputs", "__pycache__"]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

python_paths = [
    os.path.join(config.neptune_mlir_source_root, "src"),
    sysconfig.get_paths()["purelib"],
]
existing_python_path = config.environment.get("PYTHONPATH")
if existing_python_path:
    python_paths.append(existing_python_path)
config.environment["PYTHONPATH"] = os.pathsep.join(python_paths)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(
    (
        "%neptune_loop_plugin",
        os.path.join(
            config.neptune_mlir_obj_root,
            "libLoopTransform" + config.neptune_mlir_shared_library_suffix,
        ),
    )
)
config.substitutions.append(
    (
        "%neptune_htile_plugin",
        os.path.join(
            config.neptune_mlir_obj_root,
            "libHTileDialect" + config.neptune_mlir_shared_library_suffix,
        ),
    )
)
config.substitutions.append(
    (
        "%neptune_ta_plugin",
        os.path.join(
            config.neptune_mlir_obj_root,
            "libTADialect" + config.neptune_mlir_shared_library_suffix,
        ),
    )
)

llvm_config.add_tool_substitutions(
    ["mlir-opt", "FileCheck", "not"],
    [config.llvm_tools_dir],
)
