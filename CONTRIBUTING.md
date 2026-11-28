# Contributing to Troupekit

Thanks for helping. This page covers setup, branches, commits, pull requests and tests.

## Setup

1. Install Rust 1.89 or newer with [rustup](https://rustup.rs).
2. Clone the repository and build the workspace:

   ```sh
   git clone https://github.com/ChaoticQubit/troupekit
   cd troupekit
   cargo build --workspace
   ```

3. Optional, for the `llama` feature of `troupekit-local`: install CMake, then fetch the pinned llama.cpp source with `sh scripts/fetch-llama.sh`.

## Branches

Cut a short-lived branch from `main` named `type/short-slug`, where `type` is one of the commit types below, for example `fix/sqlite-wal-reopen` or `docs/quick-start`. One branch holds one change.

## Commits

Every commit subject follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```text
type(scope)!: subject
```

- `type` is one of `feat`, `fix`, `docs`, `chore`, `ci`, `test`, `refactor`, `perf`, `build`, `style`, `revert`.
- `scope` is optional: a crate or area in lower case, for example `fix(store-sqlite): ...`. Several scopes are separated by commas, for example `fix(local,kit): ...`.
- `!` marks a breaking change. Explain the break in the commit body.
- The subject starts right after `: ` and says what the change does.

CI runs `scripts/check-commits.sh` on every pull request and fails on any subject that does not match. Merge commits are skipped. Check your branch before you push:

```sh
sh scripts/check-commits.sh origin/main HEAD
```

## Pull requests

1. Open the pull request against `main`. A maintainer may ask you to target the current `release/0.x` branch instead.
2. Fill in the template: what changes, why, and how you tested it.
3. Keep it small. One pull request holds one change.
4. Every CI check must pass and every review conversation must be resolved before it merges.
5. A maintainer reviews and merges it.

## Tests

- Run the tests of the crates you changed: `cargo test -p <crate>`. CI does the same for the crates a pull request touches, and runs the whole workspace on release pull requests.
- Add a test with every fix and every feature. Prefer real code over mocks.
- Shell scripts under `scripts/` have a test beside them, run with `sh scripts/<name>-test.sh`.
- Dependency licenses and advisories are checked with `cargo deny check`.

## Code style

- Document public items with doc comments. The code base explains itself through doc comments on items and does not use inline comments.

## License

Troupekit is dual licensed under MIT or Apache-2.0. Unless you state otherwise, any contribution you submit for inclusion is dual licensed the same way, without any additional terms or conditions.

## Conduct

Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues as described in [SECURITY.md](SECURITY.md), never in a public issue.
