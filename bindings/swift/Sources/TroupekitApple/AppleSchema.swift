import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Converts the JSON Schema subset the Rust side emits into Apple's runtime schema.
/// Supported: object with properties, string (with enum), integer, number, boolean, array, anyOf, oneOf.
/// Input: schema JSON text. Output: a parsed node tree, or an error naming the unsupported keyword.
enum AppleSchema {
    static let maxSchemaBytes = 64 * 1024
    static let maxDepth = 32
    static let maxNodes = 4096

    indirect enum Node: Equatable {
        case object(name: String, description: String?, properties: [Property])
        case choice(name: String, description: String?, options: [Node])
        case string(values: [String]?)
        case integer
        case number
        case boolean
        case array(of: Node, min: Int?, max: Int?)
    }

    struct Property: Equatable {
        let name: String
        let description: String?
        let node: Node
        let optional: Bool
    }

    private static let known: Set<String> = [
        "type", "properties", "required", "items", "enum", "const", "anyOf", "oneOf",
        "description", "title", "minItems", "maxItems", "additionalProperties", "$schema",
    ]

    /// Parses and checks a schema without building Apple types, so CI can test it.
    /// Input: schema JSON text. Output: the node tree; throws on any unsupported shape.
    @discardableResult
    static func check(_ text: String) throws -> Node {
        guard text.utf8.count <= maxSchemaBytes else {
            throw AppleAdapterError("apple model: schema over \(maxSchemaBytes) bytes")
        }
        let value: Any
        do {
            value = try JSONSerialization.jsonObject(with: Data(text.utf8))
        } catch {
            throw AppleAdapterError("apple model: schema is not JSON")
        }
        var count = 0
        return try parse(value, name: "root", depth: 0, count: &count)
    }

    private static func parse(_ value: Any, name: String, depth: Int, count: inout Int) throws -> Node {
        count += 1
        guard depth < maxDepth, count <= maxNodes else {
            throw AppleAdapterError("apple model: schema deeper than \(maxDepth) or over \(maxNodes) nodes")
        }
        guard let node = value as? [String: Any] else {
            throw AppleAdapterError("apple model: schema node \(name) is not an object")
        }
        for key in node.keys.sorted() where !known.contains(key) {
            throw AppleAdapterError("apple model: schema keyword \(key) unsupported")
        }
        if let extra = node["additionalProperties"] as? Bool, extra {
            throw AppleAdapterError("apple model: schema keyword additionalProperties unsupported")
        }
        let description = node["description"] as? String
        if let options = (node["anyOf"] ?? node["oneOf"]) as? [Any] {
            guard !options.isEmpty else {
                throw AppleAdapterError("apple model: schema \(name) has an empty anyOf")
            }
            let parsed = try options.enumerated().map { index, option in
                try parse(option, name: "\(name).\(index)", depth: depth + 1, count: &count)
            }
            return .choice(name: name, description: description, options: parsed)
        }
        guard let type = node["type"] as? String else {
            throw AppleAdapterError("apple model: schema \(name) has no type")
        }
        switch type {
        case "object":
            guard let properties = node["properties"] as? [String: Any], !properties.isEmpty else {
                throw AppleAdapterError("apple model: schema \(name) is a free-form object")
            }
            let requiredOrder = (node["required"] as? [String] ?? []).filter { properties[$0] != nil }
            let required = Set(requiredOrder)
            let order = requiredOrder + properties.keys.filter { !required.contains($0) }.sorted()
            let parsed = try order.map { key in
                Property(
                    name: key,
                    description: (properties[key] as? [String: Any])?["description"] as? String,
                    node: try parse(properties[key] as Any, name: "\(name).\(key)", depth: depth + 1, count: &count),
                    optional: !required.contains(key)
                )
            }
            return .object(name: name, description: description, properties: parsed)
        case "string":
            if let value = node["const"] {
                guard let string = value as? String else {
                    throw AppleAdapterError("apple model: schema \(name) const must be a string")
                }
                return .string(values: [string])
            }
            if let values = node["enum"] {
                guard let strings = values as? [String], !strings.isEmpty else {
                    throw AppleAdapterError("apple model: schema \(name) enum must list strings")
                }
                return .string(values: strings)
            }
            return .string(values: nil)
        case "integer":
            return .integer
        case "number":
            return .number
        case "boolean":
            return .boolean
        case "array":
            guard let items = node["items"] else {
                throw AppleAdapterError("apple model: schema \(name) array has no items")
            }
            let item = try parse(items, name: "\(name).items", depth: depth + 1, count: &count)
            return .array(of: item, min: node["minItems"] as? Int, max: node["maxItems"] as? Int)
        default:
            throw AppleAdapterError("apple model: schema type \(type) unsupported")
        }
    }

    #if canImport(FoundationModels)
    /// Builds Apple's runtime schema from schema JSON text.
    /// Input: schema JSON text. Output: a `GenerationSchema`; throws on any unsupported shape.
    @available(iOS 26, macOS 26, *)
    static func build(_ text: String) throws -> GenerationSchema {
        try GenerationSchema(root: dynamic(try check(text)), dependencies: [])
    }

    @available(iOS 26, macOS 26, *)
    private static func dynamic(_ node: Node) -> DynamicGenerationSchema {
        switch node {
        case let .object(name, description, properties):
            return DynamicGenerationSchema(
                name: name,
                description: description,
                properties: properties.map { property in
                    DynamicGenerationSchema.Property(
                        name: property.name,
                        description: property.description,
                        schema: dynamic(property.node),
                        isOptional: property.optional
                    )
                }
            )
        case let .choice(name, description, options):
            return DynamicGenerationSchema(name: name, description: description, anyOf: options.map(dynamic))
        case let .string(values):
            return DynamicGenerationSchema(type: String.self, guides: values.map { [.anyOf($0)] } ?? [])
        case .integer:
            return DynamicGenerationSchema(type: Int.self, guides: [])
        case .number:
            return DynamicGenerationSchema(type: Double.self, guides: [])
        case .boolean:
            return DynamicGenerationSchema(type: Bool.self, guides: [])
        case let .array(item, min, max):
            return DynamicGenerationSchema(arrayOf: dynamic(item), minimumElements: min, maximumElements: max)
        }
    }
    #endif
}
