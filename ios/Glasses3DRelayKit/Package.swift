// swift-tools-version: 5.9
import PackageDescription

// Isolated package whose only purpose is to compile Glasses3DRelay.swift against
// the real Meta Wearables DAT SDK. Pinned to the same 0.4.0 the VisionClaw
// CameraAccess target resolves, so a compile here means it compiles there.
let package = Package(
  name: "Glasses3DRelayKit",
  platforms: [.iOS(.v17)],
  products: [.library(name: "Glasses3DRelayKit", targets: ["Glasses3DRelayKit"])],
  dependencies: [
    .package(url: "https://github.com/facebook/meta-wearables-dat-ios", exact: "0.4.0")
  ],
  targets: [
    .target(name: "Glasses3DRelayKit", dependencies: [
      .product(name: "MWDATCore", package: "meta-wearables-dat-ios"),
      .product(name: "MWDATCamera", package: "meta-wearables-dat-ios"),
    ])
  ]
)
