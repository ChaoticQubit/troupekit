package dev.troupekit.demo

import android.app.Activity
import android.app.AlertDialog
import android.graphics.Typeface
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * The demo's single screen, built from platform views: conversation, streaming answer,
 * run id, step count, window progress, and the input row. Agent wiring lands in the next M5 step;
 * the launch extras `resetState`, `scripted` and `resumeForBudgetMs` drive the end-to-end tests.
 */
class MainActivity : Activity() {
    private lateinit var state: ChatState
    private lateinit var vault: KeyVault
    private lateinit var transcript: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var status: LinearLayout
    private lateinit var question: EditText
    private lateinit var send: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "Support"
        state = ChatState(PrefsRunIdStore(applicationContext))
        vault = KeyVault(applicationContext)
        setContentView(buildLayout())
        state.onChange = { render() }
        if (intent.getBooleanExtra("resetState", false)) {
            state.reset()
        }
        render()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menu.add(Menu.NONE, MENU_KEY, Menu.NONE, "Key").setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId != MENU_KEY) return super.onOptionsItemSelected(item)
        showKeyDialog()
        return true
    }

    private fun onSend(text: String) {
        state.fail("The agent is not wired yet.", resumable = false)
    }

    private fun buildLayout(): View {
        val pad = dp(12)
        transcript = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad, pad, pad) }
        scroll = ScrollView(this).apply { addView(transcript) }
        status = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(pad, dp(4), pad, dp(4)) }
        question = EditText(this).apply {
            id = R.id.question
            hint = "Ask support"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            maxLines = 4
        }
        send = Button(this).apply {
            id = R.id.send
            text = "Send"
            setOnClickListener {
                val text = question.text.toString().trim()
                if (text.isNotEmpty()) {
                    question.setText("")
                    onSend(text)
                }
            }
        }
        val input = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(pad, pad, pad, pad)
            addView(question, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(send, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true
            addView(scroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
            addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
            addView(input, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        }
    }

    private fun render() {
        transcript.removeAllViews()
        for (line in state.lines) {
            transcript.addView(bubble(line.text, line.author, if (line.author == Author.AGENT) R.id.answer else View.NO_ID))
        }
        if (state.isRunning) {
            transcript.addView(bubble(state.streaming.ifEmpty { "…" }, Author.AGENT, R.id.streaming))
        }
        status.removeAllViews()
        state.runId?.let { runId ->
            status.addView(caption("run $runId", R.id.runId))
            status.addView(caption("steps ${state.steps}", R.id.steps))
            state.progress?.let { status.addView(caption(it, R.id.progress)) }
        }
        send.isEnabled = !state.isRunning
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun bubble(text: String, author: Author, viewId: Int): View {
        val view = TextView(this).apply {
            id = viewId
            this.text = text
            setTextIsSelectable(true)
            setPadding(dp(10), dp(8), dp(10), dp(8))
            if (author == Author.NOTICE) setTypeface(typeface, Typeface.ITALIC)
        }
        val params = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply {
            gravity = if (author == Author.USER) Gravity.END else Gravity.START
            topMargin = dp(4)
        }
        view.layoutParams = params
        return view
    }

    private fun caption(text: String, viewId: Int): View = TextView(this).apply {
        id = viewId
        this.text = text
        typeface = Typeface.MONOSPACE
        textSize = 11f
        setPadding(0, 0, dp(12), 0)
    }

    private fun showKeyDialog() {
        val field = EditText(this).apply {
            hint = "API key"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        AlertDialog.Builder(this)
            .setTitle("Cloud API key")
            .setView(field)
            .setPositiveButton("Save") { _, _ ->
                if (!vault.save(field.text.toString())) {
                    state.fail("The key could not be saved.", resumable = true)
                }
            }
            .setNeutralButton("Remove saved key") { _, _ -> vault.delete() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private companion object {
        const val MENU_KEY = 1
    }
}
