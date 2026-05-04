// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for the v0.3.2 symlink-resolution fix in
// `Sources/VLLMBridge/Bridge.swift`. The bridge calls
// `URL.resolvingSymlinksInPath()` before handing the URL to MLX so the
// MLX qwen3 codepath doesn't reject symlinked dirs with
// "Unsupported model type: qwen3". Reported by @defilan via
// defilantech/LLMKube#393 (2026-05-04).
//
// We test the Foundation URL contract directly rather than wiring up a
// full MLX model load, so the test runs in <1ms and doesn't need a
// model on disk. The behavior under test is purely:
//   "did we apply resolvingSymlinksInPath, and does it preserve canonical
//    paths while resolving symlinked ones?"

import XCTest
import Foundation

final class SymlinkResolutionTests: XCTestCase {
    var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = URL(
            fileURLWithPath: NSTemporaryDirectory(),
            isDirectory: true
        ).appendingPathComponent("vllm-swift-symlink-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpDir, withIntermediateDirectories: true
        )
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    /// `URL.resolvingSymlinksInPath()` is the API the v0.3.2 fix uses.
    /// On a path with no symlinks it must return an equivalent URL so
    /// existing canonical-path callers don't regress.
    func testCanonicalPathIsUnchanged() throws {
        let real = tmpDir.appendingPathComponent("canonical")
        try FileManager.default.createDirectory(
            at: real, withIntermediateDirectories: true
        )
        // Write a sentinel file so the dir actually exists in the FS.
        try "x".write(
            to: real.appendingPathComponent("sentinel.txt"),
            atomically: true, encoding: .utf8
        )

        let resolved = real.resolvingSymlinksInPath()
        // standardizedFileURL strips trailing slash differences and
        // resolves '..'; the canonical path's resolved form must match
        // its standardized form.
        XCTAssertEqual(
            resolved.standardizedFileURL,
            real.standardizedFileURL,
            "canonical path must round-trip through resolvingSymlinksInPath()"
        )
    }

    /// The actual reproducer from @defilan's bug report:
    /// `~/models/mlx-community/Qwen3.6-... -> ~/models/Qwen3.6-...`
    /// After resolving, the URL must point at the symlink target.
    func testSymlinkedPathResolvesToTarget() throws {
        // Set up: real dir + a symlink pointing at it.
        let target = tmpDir.appendingPathComponent("real-model")
        try FileManager.default.createDirectory(
            at: target, withIntermediateDirectories: true
        )
        try "{\"model_type\": \"qwen3_6\"}".write(
            to: target.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )

        let symlinkParent = tmpDir.appendingPathComponent("mlx-community")
        try FileManager.default.createDirectory(
            at: symlinkParent, withIntermediateDirectories: true
        )
        let symlink = symlinkParent.appendingPathComponent("Qwen3.6-35B")
        try FileManager.default.createSymbolicLink(
            at: symlink, withDestinationURL: target
        )

        // Sanity: the symlink itself exists and points where we think.
        let resolvedDestination = try FileManager.default
            .destinationOfSymbolicLink(atPath: symlink.path)
        XCTAssertEqual(resolvedDestination, target.path)

        // The behavior under test: resolvingSymlinksInPath on the symlink
        // URL produces a URL whose path equals the target's path.
        let resolved = symlink.resolvingSymlinksInPath().standardizedFileURL
        XCTAssertEqual(
            resolved.path, target.standardizedFileURL.path,
            "symlinked URL must resolve to the canonical target path"
        )

        // The original URL must still contain the symlink's name in its
        // path components — ensures we're not just no-oping above.
        XCTAssertTrue(
            symlink.path.contains("mlx-community"),
            "test setup invariant: symlinked path goes through mlx-community"
        )
        XCTAssertFalse(
            resolved.path.contains("mlx-community"),
            "resolved path must escape the symlinked parent"
        )
    }

    /// `resolvingSymlinksInPath()` is documented as returning the URL
    /// unchanged when the path doesn't exist. We rely on that so the
    /// downstream "file not found" error from MLX still surfaces with
    /// the user's original path string.
    func testNonexistentPathIsUnchanged() throws {
        let nonexistent = tmpDir.appendingPathComponent("does-not-exist")
        let resolved = nonexistent.resolvingSymlinksInPath()
        XCTAssertEqual(
            resolved.standardizedFileURL.path,
            nonexistent.standardizedFileURL.path,
            "nonexistent paths must round-trip through resolvingSymlinksInPath()"
        )
    }

    /// Nested symlinks (a -> b -> c) must collapse to the final target.
    /// Guard against any future change that accidentally only resolves
    /// one level of symlink.
    func testNestedSymlinkChainResolvesToFinalTarget() throws {
        let final = tmpDir.appendingPathComponent("final-target")
        try FileManager.default.createDirectory(
            at: final, withIntermediateDirectories: true
        )
        let mid = tmpDir.appendingPathComponent("mid-link")
        try FileManager.default.createSymbolicLink(
            at: mid, withDestinationURL: final
        )
        let outer = tmpDir.appendingPathComponent("outer-link")
        try FileManager.default.createSymbolicLink(
            at: outer, withDestinationURL: mid
        )

        let resolved = outer.resolvingSymlinksInPath().standardizedFileURL
        XCTAssertEqual(
            resolved.path, final.standardizedFileURL.path,
            "two-level symlink chain must resolve to the final target"
        )
    }
}
