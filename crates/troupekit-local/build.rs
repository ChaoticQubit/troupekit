#[path = "build/plan.rs"]
mod plan;

use plan::{decide, Link, Plan, Probe};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const ENVS: [&str; 9] = [
    "TROUPEKIT_LLAMA_SRC",
    "TROUPEKIT_LLAMA_LIB_DIR",
    "TROUPEKIT_LLAMA_METAL",
    "ANDROID_NDK_HOME",
    "CMAKE",
    "CMAKE_GENERATOR",
    "CMAKE_TOOLCHAIN_FILE",
    "TARGET_CC",
    "TARGET_CXX",
];

/// Builds or links llama.cpp and the `tk_llama` shim when the `llama`
/// feature is on; does nothing otherwise, so the default build runs no
/// C or C++ compiler. Fails with one line naming the fix.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/plan.rs");
    println!("cargo:rerun-if-changed=shim");
    for name in ENVS {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for tool in ["CC", "CXX"] {
        println!("cargo:rerun-if-env-changed={}", target_var(tool));
    }
    let probe = probe();
    match decide(&probe) {
        Ok(Plan::Skip) => {}
        Ok(Plan::Prebuilt { lib_dir, link }) => emit_link(&lib_dir, &link),
        Ok(Plan::Build {
            src,
            cmake_args,
            link,
        }) => {
            println!(
                "cargo:rerun-if-changed={}",
                src.join("include/llama.h").display()
            );
            let lib_dir = build(&cmake_args);
            emit_link(&lib_dir, &link);
        }
        Err(message) => {
            println!("cargo:warning={message}");
            panic!("{message}");
        }
    }
}

fn probe() -> Probe {
    let manifest =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let src = env::var_os("TROUPEKIT_LLAMA_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("llama.cpp"));
    let var = |name: &str| env::var(name).unwrap_or_default();
    let llama = env::var_os("CARGO_FEATURE_LLAMA").is_some();
    Probe {
        llama,
        vulkan: env::var_os("CARGO_FEATURE_LLAMA_VULKAN").is_some(),
        lib_dir: env::var_os("TROUPEKIT_LLAMA_LIB_DIR").map(PathBuf::from),
        src_exists: src.join("CMakeLists.txt").is_file() && src.join("include/llama.h").is_file(),
        src,
        cmake_found: llama && cmake_found(),
        target_os: var("CARGO_CFG_TARGET_OS"),
        target_arch: var("CARGO_CFG_TARGET_ARCH"),
        target_env: var("CARGO_CFG_TARGET_ENV"),
        target_abi: var("CARGO_CFG_TARGET_ABI"),
        host_is_target: var("HOST") == var("TARGET"),
        metal_env: env::var("TROUPEKIT_LLAMA_METAL").ok(),
        android_ndk: env::var_os("ANDROID_NDK_HOME").map(PathBuf::from),
        cmake_generator: env::var("CMAKE_GENERATOR").ok().filter(|g| !g.is_empty()),
        toolchain_file: env::var_os("CMAKE_TOOLCHAIN_FILE")
            .filter(|f| !f.is_empty())
            .map(PathBuf::from),
        target_cc: target_tool("CC"),
        target_cxx: target_tool("CXX"),
    }
}

fn target_var(tool: &str) -> String {
    format!(
        "{tool}_{}",
        env::var("TARGET").unwrap_or_default().replace('-', "_")
    )
}

fn target_tool(tool: &str) -> Option<String> {
    env::var(target_var(tool))
        .or_else(|_| env::var(format!("TARGET_{tool}")))
        .ok()
        .filter(|v| !v.is_empty())
}

fn cmake_program() -> String {
    env::var("CMAKE").unwrap_or_else(|_| "cmake".to_string())
}

fn cmake_found() -> bool {
    Command::new(cmake_program())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn build(cmake_args: &[String]) -> PathBuf {
    let manifest =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let build_dir = out.join("build");
    let lib_dir = out.join("lib");
    let mut configure = Command::new(cmake_program());
    configure
        .arg("-S")
        .arg(manifest.join("shim"))
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DTK_LIB_DIR={}", lib_dir.display()))
        .args(cmake_args);
    run(&mut configure, "cmake configure");
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "4".to_string());
    let mut compile = Command::new(cmake_program());
    compile
        .arg("--build")
        .arg(&build_dir)
        .args(["--config", "Release", "--parallel", &jobs]);
    run(&mut compile, "cmake build");
    lib_dir
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("{what} could not start: {e}"));
    assert!(
        status.success(),
        "{what} failed with {status}; see the cmake output above"
    );
}

fn emit_link(lib_dir: &Path, link: &Link) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in &link.static_libs {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    for lib in &link.system_libs {
        println!("cargo:rustc-link-lib={lib}");
    }
    for fw in &link.frameworks {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}
