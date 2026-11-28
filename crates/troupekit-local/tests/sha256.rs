use troupekit_local::sha256::{digest, to_hex, Sha256};

#[test]
fn sha256_matches_nist_vectors() {
    let cases: [(&[u8], &str); 3] = [
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
    ];
    for (input, want) in cases {
        assert_eq!(to_hex(&digest(input)), want);
    }

    let mut h = Sha256::new();
    let block = [b'a'; 1_000];
    for _ in 0..1_000 {
        h.update(&block);
    }
    assert_eq!(
        to_hex(&h.finish()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn streaming_in_odd_chunks_equals_one_shot() {
    let data: Vec<u8> = (0..1_000u32).map(|i| (i * 31 % 251) as u8).collect();
    let want = digest(&data);
    for size in [1, 3, 55, 56, 63, 64, 65, 127, 999] {
        let mut h = Sha256::new();
        for chunk in data.chunks(size) {
            h.update(chunk);
        }
        assert_eq!(h.finish(), want, "chunk size {size}");
    }
}
