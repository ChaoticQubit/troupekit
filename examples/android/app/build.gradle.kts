plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.troupekit.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.troupekit.demo"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.4.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
        getByName("test").java.srcDirs("src/test/kotlin")
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation("dev.troupekit:troupekit-android")
    testImplementation(kotlin("test"))
}
