# Wire Protocol

This document is generated from the Pydantic protocol models in `src/my_claude/core/bus`.

## Transport

- Transport: TCP
- Encoding: UTF-8
- Framing: NDJSON. Each request and response is one JSON value followed by `\n`.
- JSON-RPC version: `2.0`
- Maximum request size: configured by `max_request_bytes`; default is `65536` bytes.

## JSON-RPC Envelope

### `JsonRpcRequest`

```json
{
  "additionalProperties": false,
  "properties": {
    "jsonrpc": {
      "const": "2.0",
      "default": "2.0",
      "title": "Jsonrpc",
      "type": "string"
    },
    "id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Id"
    },
    "method": {
      "minLength": 1,
      "title": "Method",
      "type": "string"
    },
    "params": {
      "anyOf": [
        {
          "additionalProperties": true,
          "type": "object"
        },
        {
          "items": {},
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Params"
    }
  },
  "required": [
    "method"
  ],
  "title": "JsonRpcRequest",
  "type": "object"
}
```

### `JsonRpcSuccessResponse`

```json
{
  "additionalProperties": false,
  "properties": {
    "jsonrpc": {
      "const": "2.0",
      "default": "2.0",
      "title": "Jsonrpc",
      "type": "string"
    },
    "id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Id"
    },
    "result": {
      "title": "Result"
    }
  },
  "required": [
    "id",
    "result"
  ],
  "title": "JsonRpcSuccessResponse",
  "type": "object"
}
```

### `JsonRpcErrorResponse`

```json
{
  "$defs": {
    "JsonRpcError": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "title": "Code",
          "type": "integer"
        },
        "message": {
          "title": "Message",
          "type": "string"
        },
        "data": {
          "anyOf": [
            {},
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Data"
        }
      },
      "required": [
        "code",
        "message"
      ],
      "title": "JsonRpcError",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "properties": {
    "jsonrpc": {
      "const": "2.0",
      "default": "2.0",
      "title": "Jsonrpc",
      "type": "string"
    },
    "id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Id"
    },
    "error": {
      "$ref": "#/$defs/JsonRpcError"
    }
  },
  "required": [
    "id",
    "error"
  ],
  "title": "JsonRpcErrorResponse",
  "type": "object"
}
```

### `JsonRpcError`

```json
{
  "additionalProperties": false,
  "properties": {
    "code": {
      "title": "Code",
      "type": "integer"
    },
    "message": {
      "title": "Message",
      "type": "string"
    },
    "data": {
      "anyOf": [
        {},
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Data"
    }
  },
  "required": [
    "code",
    "message"
  ],
  "title": "JsonRpcError",
  "type": "object"
}
```

## Error Codes

| Name | Code |
| --- | ---: |
| `PARSE_ERROR` | `-32700` |
| `INVALID_REQUEST` | `-32600` |
| `METHOD_NOT_FOUND` | `-32601` |
| `INVALID_PARAMS` | `-32602` |
| `INTERNAL_ERROR` | `-32603` |
| `SERVER_ERROR` | `-32000` |

## Commands

### `core.ping`

S0 health check command.

**Params**

### `CorePingParams`

```json
{
  "additionalProperties": false,
  "properties": {},
  "title": "CorePingParams",
  "type": "object"
}
```

**Command Model**

### `CorePingCommand`

```json
{
  "$defs": {
    "CorePingParams": {
      "additionalProperties": false,
      "properties": {},
      "title": "CorePingParams",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "properties": {
    "method": {
      "const": "core.ping",
      "default": "core.ping",
      "title": "Method",
      "type": "string"
    },
    "params": {
      "$ref": "#/$defs/CorePingParams"
    }
  },
  "title": "CorePingCommand",
  "type": "object"
}
```

**Result**

### `CorePingResult`

```json
{
  "additionalProperties": false,
  "properties": {
    "pong": {
      "const": "pong",
      "default": "pong",
      "title": "Pong",
      "type": "string"
    },
    "uptime_seconds": {
      "minimum": 0,
      "title": "Uptime Seconds",
      "type": "number"
    },
    "server_version": {
      "title": "Server Version",
      "type": "string"
    }
  },
  "required": [
    "uptime_seconds",
    "server_version"
  ],
  "title": "CorePingResult",
  "type": "object"
}
```

**Request Example**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "core.ping",
  "params": {}
}
```

**Success Response Example**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "ok": true,
    "message": "pong"
  }
}
```

**Error Response Example**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "method not found: unknown.method"
  }
}
```
