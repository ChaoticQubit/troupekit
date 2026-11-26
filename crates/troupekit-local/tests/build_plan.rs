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
