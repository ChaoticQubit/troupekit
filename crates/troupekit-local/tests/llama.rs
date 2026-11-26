#[test]
fn llama_backend_initializes() {
    let info = troupekit_local::llama::system_info();
    assert!(!info.is_empty(), "llama.cpp reported no system info");
    assert_eq!(troupekit_local::llama::system_info(), info);
}
