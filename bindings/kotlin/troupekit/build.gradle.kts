plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val repo: File = rootDir.resolve("../..").canonicalFile

android {
    namespace = "dev.troupekit"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
        getByName("test").java.srcDirs("src/test/kotlin", "src/sharedTest/kotlin")
        getByName("androidTest") {
            java.srcDirs("src/androidTest/kotlin", "src/sharedTest/kotlin")
            assets.srcDirs(
                repo.resolve("crates/troupekit-ffi-core/contract"),
                repo.resolve("crates/troupekit-py/tests/fixtures"),
            )
        }
    }

    testOptions {
        unitTests.all {
            it.systemProperty("troupekit.repo", repo.path)
            it.systemProperty("jna.library.path", repo.resolve("target/release").path)
        }
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    api("net.java.dev.jna:jna:5.19.1@aar")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    testImplementation("net.java.dev.jna:jna:5.19.1")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
