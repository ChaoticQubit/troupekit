#![allow(dead_code)]

use std::path::PathBuf;

const IOS_MIN: &str = "16.4";
const ANDROID_PLATFORM: &str = "android-28";

/// Everything the build script learned about its environment: features,
/// override variables, whether the llama.cpp source and `cmake` exist,
/// and the target triple parts. `decide` turns it into a `Plan`.
#[derive(Debug, Clone)]
pub struct Probe {
    pub llama: bool,
    pub vulkan: bool,
    pub lib_dir: Option<PathBuf>,
    pub src: PathBuf,
    pub src_exists: bool,
    pub cmake_found: bool,
    pub target_os: String,
    pub target_arch: String,
    pub target_env: String,
    pub target_abi: String,
    pub host_is_target: bool,
    pub metal_env: Option<String>,
    pub android_ndk: Option<PathBuf>,
    pub cmake_generator: Option<String>,
    pub toolchain_file: Option<PathBuf>,
    pub target_cc: Option<String>,
    pub target_cxx: Option<String>,
}

/// Libraries the final binary links: static archives in link order,
/// system libraries the linker finds itself and Apple frameworks.
/// Printed as `cargo:rustc-link-lib` lines.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Link {
    pub static_libs: Vec<String>,
    pub system_libs: Vec<String>,
    pub frameworks: Vec<String>,
}

/// What the build script does: nothing without the `llama` feature,
/// link a prebuilt library folder, or configure and build with `cmake`
/// using `cmake_args` and then link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    Skip,
    Prebuilt {
        lib_dir: PathBuf,
        link: Link,
    },
    Build {
        src: PathBuf,
        cmake_args: Vec<String>,
        link: Link,
    },
}

/// Decides the build from the probe results, with no side effects.
/// Input: the probe. Output: the plan, or one line naming the fix
/// (the fetch script and both override variables) when a tool is missing.
pub fn decide(p: &Probe) -> Result<Plan, String> {
    if !p.llama {
        return Ok(Plan::Skip);
    }
    let apple = p.target_os == "macos" || p.target_os == "ios";
    let metal = apple && p.metal_env.as_deref() != Some("0");
    let link = link_for(p, metal);
    if let Some(dir) = &p.lib_dir {
        return Ok(Plan::Prebuilt {
            lib_dir: dir.clone(),
            link,
        });
    }
    if !p.src_exists {
        return Err(format!(
            "llama.cpp source not found at {}: run scripts/fetch-llama.sh, or set TROUPEKIT_LLAMA_SRC to a llama.cpp checkout, or set TROUPEKIT_LLAMA_LIB_DIR to a folder with a prebuilt libllama and tk_llama",
            p.src.display()
        ));
    }
    if !p.cmake_found {
        return Err(
            "cmake not found on PATH: install cmake 3.14 or newer (TROUPEKIT_LLAMA_SRC picks the llama.cpp source), or set TROUPEKIT_LLAMA_LIB_DIR to a folder with a prebuilt libllama and tk_llama"
                .to_string(),
        );
    }
    let mut args = vec![
        format!("-DTK_LLAMA_SRC={}", p.src.display()),
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DBUILD_SHARED_LIBS=OFF".to_string(),
        "-DLLAMA_BUILD_COMMON=OFF".to_string(),
        "-DLLAMA_BUILD_TESTS=OFF".to_string(),
        "-DLLAMA_BUILD_TOOLS=OFF".to_string(),
        "-DLLAMA_BUILD_EXAMPLES=OFF".to_string(),
        "-DLLAMA_BUILD_SERVER=OFF".to_string(),
        "-DLLAMA_BUILD_APP=OFF".to_string(),
        "-DLLAMA_OPENSSL=OFF".to_string(),
        "-DGGML_OPENMP=OFF".to_string(),
        format!("-DGGML_METAL={}", on_off(metal)),
        "-DGGML_METAL_EMBED_LIBRARY=ON".to_string(),
        format!("-DGGML_VULKAN={}", on_off(p.vulkan)),
    ];
    if !p.host_is_target {
        args.push("-DGGML_NATIVE=OFF".to_string());
    }
    match p.target_os.as_str() {
        "macos" => args.push(format!(
            "-DCMAKE_OSX_ARCHITECTURES={}",
            apple_arch(&p.target_arch)
        )),
        "ios" => {
            args.push("-DCMAKE_SYSTEM_NAME=iOS".to_string());
            args.push(format!(
                "-DCMAKE_OSX_ARCHITECTURES={}",
                apple_arch(&p.target_arch)
            ));
            args.push(format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={IOS_MIN}"));
            if p.target_abi == "sim" {
                args.push("-DCMAKE_OSX_SYSROOT=iphonesimulator".to_string());
            }
        }
        "android" => {
            let Some(ndk) = &p.android_ndk else {
                return Err("Android build needs the NDK: set ANDROID_NDK_HOME to the NDK root, or set TROUPEKIT_LLAMA_LIB_DIR to a folder with a prebuilt libllama and tk_llama".to_string());
            };
            args.push(format!(
                "-DCMAKE_TOOLCHAIN_FILE={}/build/cmake/android.toolchain.cmake",
                ndk.display()
            ));
            args.push(format!("-DANDROID_ABI={}", android_abi(&p.target_arch)?));
            args.push(format!("-DANDROID_PLATFORM={ANDROID_PLATFORM}"));
            args.push("-DGGML_LLAMAFILE=OFF".to_string());
            args.push("-DANDROID_STL=c++_static".to_string());
        }
        "windows" if p.target_env == "msvc" => {
            let platform = msvc_platform(&p.target_arch)?;
            let visual_studio = p
                .cmake_generator
                .as_deref()
                .map_or(true, |g| g.starts_with("Visual Studio"));
            if visual_studio {
                args.push("-A".to_string());
                args.push(platform.to_string());
            } else if !p.host_is_target {
                cross_toolchain(p, &mut args)?;
            }
        }
        _ if !p.host_is_target => cross_toolchain(p, &mut args)?,
        _ => {}
    }
    Ok(Plan::Build {
        src: p.src.clone(),
        cmake_args: args,
        link,
    })
}

fn msvc_platform(arch: &str) -> Result<&'static str, String> {
    match arch {
        "x86_64" => Ok("x64"),
        "aarch64" => Ok("ARM64"),
        "x86" => Ok("Win32"),
        other => Err(format!(
            "Windows target arch {other} has no Visual Studio platform; build for x86_64, aarch64 or x86"
        )),
    }
}

fn cross_toolchain(p: &Probe, args: &mut Vec<String>) -> Result<(), String> {
    if let Some(file) = &p.toolchain_file {
        args.push(format!("-DCMAKE_TOOLCHAIN_FILE={}", file.display()));
        return Ok(());
    }
    let system = match p.target_os.as_str() {
        "linux" => "Linux",
        "windows" => "Windows",
        "freebsd" => "FreeBSD",
        "netbsd" => "NetBSD",
        "openbsd" => "OpenBSD",
        other => {
            return Err(format!(
                "cross-building llama.cpp for {other} needs CMAKE_TOOLCHAIN_FILE"
            ))
        }
    };
    let (Some(cc), Some(cxx)) = (&p.target_cc, &p.target_cxx) else {
        return Err(format!(
            "cross-building llama.cpp for {} {} needs the target compilers: set CC_{arch}_unknown_{os}_{env} and CXX_..., or TARGET_CC and TARGET_CXX, or CMAKE_TOOLCHAIN_FILE (it also sets the sysroot), or TROUPEKIT_LLAMA_LIB_DIR to a folder with a prebuilt libllama and tk_llama",
            p.target_os,
            p.target_arch,
            arch = p.target_arch,
            os = p.target_os,
            env = p.target_env,
        ));
    };
    args.push(format!("-DCMAKE_SYSTEM_NAME={system}"));
    args.push(format!("-DCMAKE_SYSTEM_PROCESSOR={}", p.target_arch));
    args.push(format!("-DCMAKE_C_COMPILER={cc}"));
    args.push(format!("-DCMAKE_CXX_COMPILER={cxx}"));
    Ok(())
}

fn link_for(p: &Probe, metal: bool) -> Link {
    let apple = p.target_os == "macos" || p.target_os == "ios";
    let mut static_libs: Vec<String> = ["tk_llama", "llama", "ggml", "ggml-cpu"]
        .map(String::from)
        .to_vec();
    let mut system_libs = Vec::new();
    let mut frameworks = Vec::new();
    if metal {
        static_libs.push("ggml-metal".into());
        frameworks.extend(["Foundation", "Metal", "MetalKit"].map(String::from));
    }
    if apple {
        static_libs.push("ggml-blas".into());
        frameworks.push("Accelerate".into());
        system_libs.push("c++".into());
    }
    if p.vulkan {
        static_libs.push("ggml-vulkan".into());
        system_libs.push(
            if p.target_os == "windows" {
                "vulkan-1"
            } else {
                "vulkan"
            }
            .into(),
        );
    }
    static_libs.push("ggml-base".into());
    match p.target_os.as_str() {
        "linux" => system_libs.extend(["stdc++", "m", "pthread", "dl"].map(String::from)),
        "android" => system_libs.extend(["c++_static", "c++abi", "log", "m"].map(String::from)),
        "freebsd" | "openbsd" => system_libs.extend(["c++", "m", "pthread"].map(String::from)),
        "netbsd" => system_libs.extend(["stdc++", "m", "pthread"].map(String::from)),
        _ => {}
    }
    Link {
        static_libs,
        system_libs,
        frameworks,
    }
}

fn on_off(b: bool) -> &'static str {
    if b {
        "ON"
    } else {
        "OFF"
    }
}

fn apple_arch(arch: &str) -> &str {
    if arch == "aarch64" {
        "arm64"
    } else {
        arch
    }
}

fn android_abi(arch: &str) -> Result<&'static str, String> {
    match arch {
        "aarch64" => Ok("arm64-v8a"),
        "x86_64" => Ok("x86_64"),
        "arm" => Ok("armeabi-v7a"),
        "x86" => Ok("x86"),
        other => Err(format!(
            "Android target arch {other} has no NDK ABI name; build for aarch64 or x86_64"
        )),
    }
}
