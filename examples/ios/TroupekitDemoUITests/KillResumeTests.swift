import XCTest

/// End-to-end tests of the iOS demo on the simulator, with the scripted host model.
/// The on-device model cannot run on CI runners (plan R3), so a debug-only scripted model
/// answers in several timed steps; the app shows the run id and finished step count.
final class KillResumeTests: XCTestCase {
    private let timeout: TimeInterval = 60

    override func setUp() {
        continueAfterFailure = false
    }

    private func launch(reset: Bool, extra: [String] = []) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchEnvironment["TROUPEKIT_SCRIPTED"] = "1"
        app.launchArguments = (reset ? ["-resetState"] : []) + extra
        app.launch()
        return app
    }

    private func ask(_ app: XCUIApplication, _ question: String) {
        let field = app.textFields["question"].exists ? app.textFields["question"] : app.textViews["question"]
        XCTAssertTrue(field.waitForExistence(timeout: timeout))
        field.tap()
        field.typeText(question)
        app.buttons["send"].tap()
    }

    private func waitFor(_ element: XCUIElement, label predicate: String) {
        let expectation = XCTNSPredicateExpectation(predicate: NSPredicate(format: predicate), object: element)
        XCTAssertEqual(XCTWaiter().wait(for: [expectation], timeout: timeout), .completed, "\(element) never matched \(predicate)")
    }

    func test_ios_kill_mid_run_resumes() {
        var app = launch(reset: true)
        ask(app, "How do I reset my password?")
        let steps = app.staticTexts["steps"]
        waitFor(steps, label: "label == 'steps 2'")
        let runId = app.staticTexts["runId"].label
        XCTAssertTrue(runId.hasPrefix("run "), runId)
        app.terminate()

        app = launch(reset: false)
        XCTAssertTrue(app.staticTexts["runId"].waitForExistence(timeout: timeout))
        XCTAssertEqual(app.staticTexts["runId"].label, runId)
        let answers = app.staticTexts.matching(identifier: "answer")
        waitFor(answers.firstMatch, label: "label CONTAINS 'account-password-reset.md'")
        XCTAssertEqual(answers.count, 1)
    }

    func test_ios_stream_shows_deltas() {
        let app = launch(reset: true)
        ask(app, "How do I reset my password?")
        let streaming = app.staticTexts["streaming"]
        XCTAssertTrue(streaming.waitForExistence(timeout: timeout))
        waitFor(streaming, label: "label CONTAINS 'Open Settings'")
        let partial = streaming.label
        let answers = app.staticTexts.matching(identifier: "answer")
        waitFor(answers.firstMatch, label: "label CONTAINS 'account-password-reset.md'")
        XCTAssertTrue(answers.firstMatch.label.hasPrefix(partial), "\(partial) is not a prefix of the answer")
        XCTAssertLessThan(partial.count, answers.firstMatch.label.count)
    }

    func test_ios_resume_for_returns_pending_on_zero_budget() {
        let app = launch(reset: true, extra: ["-resumeForBudgetMs", "0"])
        ask(app, "How do I reset my password?")
        let progress = app.staticTexts["progress"]
        XCTAssertTrue(progress.waitForExistence(timeout: timeout))
        waitFor(progress, label: "label == 'pending'")
        XCTAssertEqual(app.staticTexts.matching(identifier: "answer").count, 0)
    }
}
