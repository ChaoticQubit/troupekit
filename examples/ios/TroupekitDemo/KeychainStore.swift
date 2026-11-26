import Foundation
import Security

/// Keeps the user-typed cloud API key in the Keychain, never in the bundle or defaults.
/// Items are this-device-only and readable only while the device is unlocked.
/// Input: a service name. Output: save, load and delete of one key string.
struct KeychainStore {
    static let maxKeyBytes = 4096

    let service: String
    let account = "api-key"

    /// Stores the key, replacing any earlier one.
    /// Input: the key text, 1 to `maxKeyBytes` bytes. Output: true when the Keychain accepted it.
    func save(_ key: String) -> Bool {
        let data = Data(key.utf8)
        guard !data.isEmpty, data.count <= Self.maxKeyBytes else { return false }
        delete()
        var item = query()
        item[kSecValueData as String] = data
        item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        return SecItemAdd(item as CFDictionary, nil) == errSecSuccess
    }

    /// Reads the stored key.
    /// Input: none. Output: the key, or nil when none is stored or it is unreadable.
    func load() -> String? {
        var item = query()
        item[kSecReturnData as String] = true
        item[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        guard SecItemCopyMatching(item as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data,
              data.count <= Self.maxKeyBytes
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Removes the stored key.
    /// Input: none. Output: none; missing items are not an error.
    func delete() {
        SecItemDelete(query() as CFDictionary)
    }

    private func query() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }
}
