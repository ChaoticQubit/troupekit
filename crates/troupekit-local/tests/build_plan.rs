use crate::build_plan_src::{decide, Plan, Probe};
use std::path::PathBuf;

fn probe() -> Probe {
    Probe {
        llama: true,
        vulkan: false,
        lib_dir: None,
        src: PathBuf::from("/x/llama.cpp"),
        src_exists: true,
        cmake_found: true,
        target_os: "macos".into(),
        target_arch: "aarch64".into(),
        target_env: String::new(),
        target_abi: String::new(),
        host_is_target: true,
        metal_env: None,
        android_ndk: None,
        cmake_generator: None,
        toolchain_file: None,
        target_cc: None,
        target_cxx: None,
    }
}

fn build_args(p: &Probe) -> Vec<String> {
    match decide(p).expect("plan") {
        Plan::Build { cmake_args, .. } => cmake_args,
        other => panic!("expected a build, got {other:?}"),
    }
}

#[test]
fn missing_source_or_cmake_error_names_the_fix() {
    let mut p = probe();
    p.src_exists = false;
    let e = decide(&p).expect_err("no source");
    assert!(e.contains("scripts/fetch-llama.sh"), "{e}");
    assert!(
        e.contains("TROUPEKIT_LLAMA_SRC") && e.contains("TROUPEKIT_LLAMA_LIB_DIR"),
        "{e}"
    );
    assert!(e.contains("/x/llama.cpp"), "{e}");
    assert!(!e.contains('\n'), "one line: {e}");

    let mut p = probe();
    p.cmake_found = false;
    let e = decide(&p).expect_err("no cmake");
    assert!(e.contains("cmake"), "{e}");
    assert!(
        e.contains("TROUPEKIT_LLAMA_SRC") && e.contains("TROUPEKIT_LLAMA_LIB_DIR"),
        "{e}"
    );
    assert!(!e.contains('\n'), "one line: {e}");
}

#[test]
fn feature_off_skips_everything() {
    let mut p = probe();
    p.llama = false;
    p.src_exists = false;
    p.cmake_found = false;
    assert_eq!(decide(&p).expect("skip"), Plan::Skip);
}

#[test]
fn lib_dir_skips_the_build() {
    let mut p = probe();
    p.lib_dir = Some(PathBuf::from("/prebuilt"));
    p.src_exists = false;
    p.cmake_found = false;
    match decide(&p).expect("prebuilt") {
        Plan::Prebuilt { lib_dir, link } => {
            assert_eq!(lib_dir, PathBuf::from("/prebuilt"));
            assert!(link.static_libs.contains(&"llama".to_string()));
        }
        other => panic!("expected prebuilt, got {other:?}"),
    }
}

#[test]
fn apple_targets_use_metal_unless_turned_off() {
    let args = build_args(&probe());
    assert!(args.contains(&"-DGGML_METAL=ON".to_string()), "{args:?}");
    assert!(args.contains(&"-DGGML_METAL_EMBED_LIBRARY=ON".to_string()));

    let mut p = probe();
    p.metal_env = Some("0".into());
    let args = build_args(&p);
    assert!(args.contains(&"-DGGML_METAL=OFF".to_string()), "{args:?}");

    let mut p = probe();
    p.target_os = "linux".into();
    p.target_arch = "x86_64".into();
    p.target_env = "gnu".into();
    let args = build_args(&p);
    assert!(args.contains(&"-DGGML_METAL=OFF".to_string()), "{args:?}");
}

#[test]
fn default_build_turns_off_extras_and_network() {
    let args = build_args(&probe());
    for want in [
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_COMMON=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_OPENSSL=OFF",
        "-DGGML_OPENMP=OFF",
        "-DGGML_VULKAN=OFF",
    ] {
        assert!(args.contains(&want.to_string()), "missing {want}: {args:?}");
    }
}

#[test]
fn ios_and_android_cross_builds_set_their_toolchain() {
    let mut p = probe();
    p.target_os = "ios".into();
    p.host_is_target = false;
    let args = build_args(&p);
    assert!(
        args.contains(&"-DCMAKE_SYSTEM_NAME=iOS".to_string()),
        "{args:?}"
    );
    assert!(args.contains(&"-DGGML_NATIVE=OFF".to_string()));

    let mut p = probe();
    p.target_os = "android".into();
    p.host_is_target = false;
    let e = decide(&p).expect_err("no NDK");
    assert!(e.contains("ANDROID_NDK_HOME"), "{e}");

    p.android_ndk = Some(PathBuf::from("/ndk"));
    let args = build_args(&p);
    assert!(
        args.contains(
            &"-DCMAKE_TOOLCHAIN_FILE=/ndk/build/cmake/android.toolchain.cmake".to_string()
        ),
        "{args:?}"
    );
    assert!(args.contains(&"-DANDROID_ABI=arm64-v8a".to_string()));
}

#[test]
fn vulkan_feature_turns_vulkan_on() {
    let mut p = probe();
    p.target_os = "windows".into();
    p.target_arch = "x86_64".into();
    p.target_env = "msvc".into();
    p.vulkan = true;
    let args = build_args(&p);
    assert!(args.contains(&"-DGGML_VULKAN=ON".to_string()), "{args:?}");
    match decide(&p).expect("plan") {
        Plan::Build { link, .. } => {
            assert!(link.static_libs.contains(&"ggml-vulkan".to_string()));
            assert!(link.system_libs.contains(&"vulkan-1".to_string()));
        }
        other => panic!("expected a build, got {other:?}"),
    }
}

fn msvc(arch: &str) -> Probe {
    let mut p = probe();
    p.target_os = "windows".into();
    p.target_env = "msvc".into();
    p.target_arch = arch.into();
    p
}

#[test]
fn msvc_platform_matches_the_target_arch() {
    let platform = |arch: &str| {
        let args = build_args(&msvc(arch));
        let at = args.iter().position(|a| a == "-A").expect("-A");
        args[at + 1].clone()
    };
    assert_eq!(platform("x86_64"), "x64");
    assert_eq!(platform("aarch64"), "ARM64");
    assert_eq!(platform("x86"), "Win32");
    assert!(decide(&msvc("riscv64")).is_err());
}

#[test]
fn ninja_generator_gets_no_platform_argument() {
    let mut p = msvc("x86_64");
    p.cmake_generator = Some("Ninja".into());
    let args = build_args(&p);
    assert!(!args.iter().any(|a| a == "-A"), "{args:?}");
    p.cmake_generator = Some("Visual Studio 17 2022".into());
    assert!(build_args(&p).iter().any(|a| a == "-A"));
    let mut cross = msvc("aarch64");
    cross.host_is_target = false;
    cross.cmake_generator = Some("Ninja".into());
    let err = decide(&cross).expect_err("ninja cannot pick the arch");
    assert!(err.contains("CMAKE_TOOLCHAIN_FILE"), "{err}");
}

#[test]
fn linux_cross_build_uses_the_target_toolchain() {
    let mut p = probe();
    p.target_os = "linux".into();
    p.target_arch = "aarch64".into();
    p.target_env = "gnu".into();
    p.host_is_target = false;
    let err = decide(&p).expect_err("no target compiler");
    assert!(
        err.contains("CC_aarch64") || err.contains("TARGET_CC"),
        "{err}"
    );
    assert!(err.contains("CMAKE_TOOLCHAIN_FILE"), "{err}");
    p.target_cc = Some("aarch64-linux-gnu-gcc".into());
    p.target_cxx = Some("aarch64-linux-gnu-g++".into());
    let args = build_args(&p);
    for want in [
        "-DCMAKE_SYSTEM_NAME=Linux",
        "-DCMAKE_SYSTEM_PROCESSOR=aarch64",
        "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc",
        "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++",
        "-DGGML_NATIVE=OFF",
    ] {
        assert!(args.contains(&want.to_string()), "{want} missing: {args:?}");
    }
    p.toolchain_file = Some(PathBuf::from("/tc/aarch64.cmake"));
    let args = build_args(&p);
    assert!(args.contains(&"-DCMAKE_TOOLCHAIN_FILE=/tc/aarch64.cmake".to_string()));
    assert!(
        !args.iter().any(|a| a.starts_with("-DCMAKE_C_COMPILER")),
        "{args:?}"
    );
    let mut native = probe();
    native.target_os = "linux".into();
    native.target_arch = "x86_64".into();
    assert!(!build_args(&native)
        .iter()
        .any(|a| a.starts_with("-DCMAKE_SYSTEM_NAME")));
}

#[test]
fn every_unix_target_links_a_cpp_runtime() {
    for (os, runtime) in [
        ("linux", "stdc++"),
        ("freebsd", "c++"),
        ("openbsd", "c++"),
        ("netbsd", "stdc++"),
    ] {
        let mut p = probe();
        p.target_os = os.into();
        p.target_arch = "x86_64".into();
        match decide(&p).expect("plan") {
            Plan::Build { link, .. } => {
                assert!(
                    link.system_libs.contains(&runtime.to_string()),
                    "{os}: {:?}",
                    link.system_libs
                );
                assert!(link.system_libs.contains(&"pthread".to_string()), "{os}");
            }
            other => panic!("{os}: expected a build, got {other:?}"),
        }
    }
}
