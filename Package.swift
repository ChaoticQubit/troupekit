// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Troupekit",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Troupekit", targets: ["Troupekit"]),
        .library(name: "TroupekitApple", targets: ["TroupekitApple"]),
    ],
    targets: [
        .binaryTarget(
            name: "TroupekitFFI",
            path: "bindings/swift/TroupekitFFI.xcframework"
        ),
        .target(
            name: "Troupekit",
            dependencies: ["TroupekitFFI"],
            path: "bindings/swift/Sources/Troupekit"
        ),
        .target(
            name: "TroupekitApple",
            dependencies: ["Troupekit"],
            path: "bindings/swift/Sources/TroupekitApple"
        ),
        .testTarget(
            name: "TroupekitTests",
            dependencies: ["Troupekit"],
            path: "bindings/swift/Tests/TroupekitTests"
        ),
        .testTarget(
            name: "TroupekitAppleTests",
            dependencies: ["TroupekitApple"],
            path: "bindings/swift/TestsApple/TroupekitAppleTests"
        ),
    ]
)
