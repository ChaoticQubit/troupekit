use std::io::Cursor;
use troupekit_trace::view::{read_jsonl, render_html, SpanTree, ViewLimits};

const SAMPLE: &str = include_str!("data/sample.jsonl");
const HOSTILE: &str = "</script><img src=x onerror=alert(1)>";

fn tree_with_hostile() -> SpanTree {
    let line = format!(
        "{{\"v\":1,\"t\":\"start\",\"trace\":\"00000000000000ab\",\"span\":9,\"parent\":1,\"cat\":\"tools\",\"name\":\"{HOSTILE}\",\"ts_us\":1700,\"fields\":{{\"{HOSTILE}\":\"{HOSTILE} &amp; \\u2028\"}}}}\n"
    );
    let text = format!("{SAMPLE}{line}");
    read_jsonl(Cursor::new(text.into_bytes()), ViewLimits::default()).unwrap()
}

fn data_block(page: &str) -> &str {
    let open = "<script type=\"application/json\" id=\"trace\">";
    let start = page.find(open).expect("data block") + open.len();
    let len = page[start..].find("</script>").expect("data block end");
    &page[start..start + len]
}

#[test]
fn html_escapes_hostile_content() {
    let page = render_html(&tree_with_hostile());
    assert!(!page.contains("<img"));
    assert!(!page.contains(HOSTILE));
    assert!(!page.contains("&amp;"));
    let data = data_block(&page);
    assert!(!data.contains(['<', '>', '&']));
    assert!(data.contains("\\u003c/script\\u003e\\u003cimg src=x onerror=alert(1)\\u003e"));
    let parsed: serde_json::Value = serde_json::from_str(data).expect("data block is JSON");
    let names: Vec<&str> = parsed["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["n"].as_str().unwrap())
        .collect();
    assert!(names.contains(&HOSTILE));
}

#[test]
fn html_has_no_external_urls() {
    let full = render_html(&tree_with_hostile());
    let data = data_block(&full);
    let page = full.replacen(data, "", 1);
    for needle in [
        "http:",
        "https:",
        "//cdn",
        "url(",
        "@import",
        " src=",
        " href=",
        "fetch(",
        "XMLHttpRequest",
    ] {
        assert!(!page.contains(needle), "page contains {needle}");
    }
    assert!(page.contains(
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'\">"
    ));
}

#[test]
fn html_embeds_every_span() {
    let tree = tree_with_hostile();
    let page = render_html(&tree);
    let parsed: serde_json::Value = serde_json::from_str(data_block(&page)).unwrap();
    let nodes = parsed["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), tree.nodes().len());
    for (node, json) in tree.nodes().iter().zip(nodes) {
        assert_eq!(json["n"], node.name());
        assert_eq!(json["c"], node.category().as_str());
    }
    assert_eq!(
        parsed["roots"].as_array().unwrap().len(),
        tree.roots().len()
    );
    assert_eq!(parsed["name"], "demo");
    assert!(
        nodes.iter().any(|n| n["e"].is_null() && n["k"] == "s"),
        "open span kept"
    );
}

#[test]
fn html_uses_text_content_only() {
    let page = render_html(&tree_with_hostile());
    for needle in [
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ] {
        assert!(!page.contains(needle), "page uses {needle}");
    }
    assert!(page.contains("textContent"));
    assert!(!page.contains("<!--"));
}
