use troupekit_core::event::Usage;
use troupekit_providers::catalog::{Catalog, CatalogError, Price};
use troupekit_providers::wire::{AuthStyle, MaxTokensField, Protocol, SchemaMode};

const CATALOG: &str = include_str!("../../../catalog/providers.toml");

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage {
        input_tokens,
        output_tokens,
        cost_usd_micros: 0,
    }
}

fn vendor(extra: &str) -> String {
    format!("[[vendor]]\nid = \"v\"\nname = \"V\"\nprotocol = \"openai_compatible\"\nbase_url = \"https://v.test/v1\"\nauth = \"bearer\"\n{extra}\n")
}

#[test]
fn shipped_catalog_loads_with_every_required_vendor() {
    let c = Catalog::load(CATALOG).unwrap();
    let ids: Vec<&str> = c.vendors().iter().map(|v| v.id.as_str()).collect();
    for want in [
        "anthropic",
        "openai",
        "google",
        "openrouter",
        "groq",
        "deepseek",
        "mistral",
        "ollama",
        "lmstudio",
    ] {
        assert!(ids.contains(&want), "{want}");
    }
    let anthropic = c.vendor("anthropic").unwrap();
    assert_eq!(
        (
            anthropic.protocol,
            anthropic.auth,
            anthropic.quirks.temperature
        ),
        (Protocol::Anthropic, AuthStyle::XApiKey, false)
    );
    for m in [
        "anthropic/claude-opus-5-5",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-haiku-5-5",
        "openai/gpt-6.2-sol",
        "openai/gpt-6.1-terra",
        "openai/gpt-6-luna",
        "google/gemini-3.8-flash",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3.1-pro-preview",
    ] {
        let price = c.price(m).unwrap();
        assert!(
            price.verified && price.output_usd_micros_per_mtok > 0,
            "{m}"
        );
    }
    for (id, source) in [
        (
            "anthropic",
            "https://platform.claude.com/docs/en/about-claude/pricing",
        ),
        ("openai", "https://developers.openai.com/api/docs/pricing"),
        ("google", "https://ai.google.dev/gemini-api/docs/pricing"),
    ] {
        assert_eq!(
            c.vendor(id).unwrap().price_source.as_deref(),
            Some(source),
            "{id}"
        );
    }
    assert_eq!(
        c.price("openai/gpt-6.2-sol")
            .unwrap()
            .cost_usd_micros(&usage(272_001, 1_000_000)),
        1_088_004 + 15_000_000
    );
    assert_eq!(
        c.price("google/gemini-3.1-pro-preview")
            .unwrap()
            .cost_usd_micros(&usage(200_000, 1_000_000)),
        400_000 + 12_000_000
    );
    assert_eq!(
        c.vendor("openai").unwrap().quirks.max_tokens_field,
        MaxTokensField::MaxCompletionTokens
    );
    assert_eq!(c.vendor("google").unwrap().protocol, Protocol::Gemini);
    assert_eq!(
        c.vendor("deepseek").unwrap().quirks.output_schema,
        SchemaMode::Unsupported
    );
    for local in ["ollama", "lmstudio"] {
        let v = c.vendor(local).unwrap();
        assert!(v.open_models && v.auth == AuthStyle::NoKey, "{local}");
    }
    for v in c.vendors() {
        for m in &v.models {
            if !m.price_verified {
                assert_eq!(
                    (m.input_usd_micros_per_mtok, m.output_usd_micros_per_mtok),
                    (0, 0),
                    "{}/{}",
                    v.id,
                    m.id
                );
            }
            assert!(m.context_window > 0, "{}/{}", v.id, m.id);
        }
    }
}

#[test]
fn prices_round_up_and_switch_to_long_context_rates() {
    let c = Catalog::load(CATALOG).unwrap();
    let opus = c.price("anthropic/claude-opus-5-5").unwrap();
    assert_eq!(
        opus.cost_usd_micros(&usage(1_000_000, 1_000_000)),
        24_000_000
    );
    assert_eq!(opus.cost_usd_micros(&usage(1, 0)), 4);
    let haiku = c.price("anthropic/claude-haiku-5-5").unwrap();
    assert_eq!(haiku.cost_usd_micros(&usage(100_000, 0)), 10_000);
    assert_eq!(haiku.cost_usd_micros(&usage(100_001, 0)), 50_001);
    assert_eq!(haiku.cost_usd_micros(&usage(1, 1)), 1);
    assert_eq!(Price::FREE.cost_usd_micros(&usage(u64::MAX, u64::MAX)), 0);
    let max = Price {
        input_usd_micros_per_mtok: u64::MAX,
        output_usd_micros_per_mtok: u64::MAX,
        verified: true,
        long_context: None,
        search_usd_micros: 0,
    };
    assert_eq!(max.cost_usd_micros(&usage(u64::MAX, u64::MAX)), u64::MAX);
}

#[test]
fn hosted_searches_are_priced_per_search() {
    use serde_json::json;
    use troupekit_core::message::{ContentPart, Message, Role};
    use troupekit_providers::request::{CompletionResponse, StopReason};
    let c = Catalog::load(CATALOG).unwrap();
    let opus = c.price("anthropic/claude-opus-5-5").unwrap();
    assert_eq!(opus.search_usd_micros, 10_000);
    assert_eq!(c.price("openai/gpt-6.2-sol").unwrap().search_usd_micros, 0);
    let call = ContentPart::Hosted {
        tool: Some("web_search".into()),
        block: json!({"type": "server_tool_use"}),
    };
    let fetch = ContentPart::Hosted {
        tool: Some("web_fetch".into()),
        block: json!({"type": "server_tool_use"}),
    };
    let reply = CompletionResponse {
        message: Message::new(Role::Assistant, vec![call.clone(), call, fetch]).unwrap(),
        usage: usage(1_000_000, 0),
        stop: StopReason::EndTurn,
    };
    assert_eq!(opus.reply_cost_usd_micros(&reply), 4_000_000 + 2 * 10_000);
}

#[test]
fn resolves_selectors() {
    let c = Catalog::load(CATALOG).unwrap();
    assert!(c.price("openrouter/anthropic/claude-opus-5.5").is_ok());
    assert_eq!(c.price("ollama/qwen3:8b").unwrap(), Price::FREE);
    assert!(matches!(
        c.price("anthropic"),
        Err(CatalogError::BadSelector(_))
    ));
    assert!(matches!(c.price("/x"), Err(CatalogError::BadSelector(_))));
    assert!(matches!(
        c.price("nobody/x"),
        Err(CatalogError::UnknownVendor(_))
    ));
    assert!(matches!(
        c.price("anthropic/claude-nope"),
        Err(CatalogError::UnknownModel(_))
    ));
    assert!(matches!(
        c.price("ollama/bad id"),
        Err(CatalogError::BadSelector(_))
    ));
}

#[test]
fn rejects_bad_catalogs() {
    let model = "[[vendor.model]]\nid = \"m\"\ncontext_window = 1\ninput_usd_micros_per_mtok = 0\noutput_usd_micros_per_mtok = 0\nprice_verified = false\n";
    assert!(Catalog::load(&vendor(model)).is_ok());
    let paid = model
        .replace(
            "output_usd_micros_per_mtok = 0",
            "output_usd_micros_per_mtok = 5",
        )
        .replace("price_verified = false", "price_verified = true");
    assert!(Catalog::load(&vendor(&format!(
        "price_source = \"https://v.test/prices\"\n{paid}"
    )))
    .is_ok());
    let parse = [
        "vendor = = 1".to_string(),
        vendor("color = \"red\""),
        vendor("quirks = { typo = true }"),
        vendor("protocol2 = 1"),
    ];
    for text in parse {
        assert!(
            matches!(Catalog::load(&text), Err(CatalogError::Parse(_))),
            "{text}"
        );
    }
    let invalid = [
        format!("{}{}", vendor(""), vendor("")),
        vendor("").replace("id = \"v\"", "id = \"Bad Id\""),
        vendor("").replace("name = \"V\"", "name = \" \""),
        vendor("").replace("https://v.test/v1", "http://v.test/v1"),
        vendor("").replace("https://v.test/v1", "ftp://v.test"),
        vendor(&model.replace("id = \"m\"", "id = \"a b\"")),
        vendor(&format!("{model}{model}")),
        vendor(&format!("{model}long_context = {{ above_input_tokens = 0, input_usd_micros_per_mtok = 1, output_usd_micros_per_mtok = 1 }}\n")),
        vendor(&paid),
        vendor(&format!("price_source = \"http://v.test/prices\"\n{paid}")),
    ];
    for text in invalid {
        assert!(
            matches!(Catalog::load(&text), Err(CatalogError::Invalid(_))),
            "{text}"
        );
    }
    let keyless_http = vendor("")
        .replace("https://v.test/v1", "http://localhost:9/v1")
        .replace("auth = \"bearer\"", "auth = \"none\"");
    assert!(Catalog::load(&keyless_http).is_ok());
}
