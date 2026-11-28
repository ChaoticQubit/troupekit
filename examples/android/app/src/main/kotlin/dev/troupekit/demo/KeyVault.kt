package dev.troupekit.demo

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Keeps the user-typed cloud API key encrypted with an AES-GCM key held in the Android Keystore.
 * Only the ciphertext and IV reach SharedPreferences; the key never leaves the Keystore, never the APK.
 * Input: the app context. Output: save, load and delete of one key string.
 */
class KeyVault(context: Context) {
    private val prefs = context.getSharedPreferences("troupekit.demo.key", Context.MODE_PRIVATE)

    /** Encrypts and stores the key. Input: 1 to MAX_KEY_CHARS characters. Output: true when stored. */
    fun save(apiKey: String): Boolean {
        if (apiKey.isEmpty() || apiKey.length > MAX_KEY_CHARS) return false
        return runCatching {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.ENCRYPT_MODE, secretKey())
            val sealed = cipher.doFinal(apiKey.toByteArray(Charsets.UTF_8))
            prefs.edit()
                .putString(DATA, Base64.encodeToString(sealed, Base64.NO_WRAP))
                .putString(IV, Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
                .commit()
        }.getOrDefault(false)
    }

    /** Decrypts the stored key. Output: the key, or null when none is stored or it cannot be opened. */
    fun load(): String? {
        val data = prefs.getString(DATA, null) ?: return null
        val iv = prefs.getString(IV, null) ?: return null
        return runCatching {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.DECRYPT_MODE, secretKey(), GCMParameterSpec(TAG_BITS, Base64.decode(iv, Base64.NO_WRAP)))
            String(cipher.doFinal(Base64.decode(data, Base64.NO_WRAP)), Charsets.UTF_8)
        }.getOrNull()
    }

    /** Removes the stored ciphertext. Output: none. */
    fun delete() {
        prefs.edit().clear().commit()
    }

    private fun secretKey(): SecretKey {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (store.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build(),
        )
        return generator.generateKey()
    }

    companion object {
        const val MAX_KEY_CHARS = 4096
        private const val KEYSTORE = "AndroidKeyStore"
        private const val ALIAS = "troupekit.demo.apikey"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val TAG_BITS = 128
        private const val DATA = "data"
        private const val IV = "iv"
    }
}
