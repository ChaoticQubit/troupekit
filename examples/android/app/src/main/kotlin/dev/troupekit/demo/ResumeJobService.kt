package dev.troupekit.demo

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context

/**
 * Background window of the demo: resumes a stored, unfinished run for at most `BUDGET_MS`.
 * Platform JobScheduler, no WorkManager. The shell finishes at once; the agent wiring adds
 * `resumeFor(runId, budgetMs)` on a worker thread and reports pending or done to the app.
 */
class ResumeJobService : JobService() {
    override fun onStartJob(params: JobParameters): Boolean {
        return false
    }

    override fun onStopJob(params: JobParameters): Boolean {
        return true
    }

    companion object {
        const val JOB_ID = 1
        const val BUDGET_MS = 25_000L

        /** Schedules the window once network is available. Input: a context. Output: true when scheduled. */
        fun schedule(context: Context): Boolean {
            val scheduler = context.getSystemService(JobScheduler::class.java) ?: return false
            val job = JobInfo.Builder(JOB_ID, ComponentName(context, ResumeJobService::class.java))
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
                .build()
            return scheduler.schedule(job) == JobScheduler.RESULT_SUCCESS
        }

        /** Cancels a scheduled window once the run is done. Input: a context. Output: none. */
        fun cancel(context: Context) {
            context.getSystemService(JobScheduler::class.java)?.cancel(JOB_ID)
        }
    }
}
