//! On-device models for Troupekit: one sync `LocalModel` trait,
//! a checked model catalog and cache, and llama.cpp behind the `llama`
//! feature. The default build compiles no C or C++ code.

use troupekit_providers as _;
use troupekit_trace as _;
