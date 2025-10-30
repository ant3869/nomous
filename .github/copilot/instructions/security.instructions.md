---
description: 'Instructions for security best practices and code quality in offline-first applications'
---

# Security & Code Quality Guidelines

## WebSocket Security

1. Input Validation
```typescript
const validateMessage = (message: unknown): MessageType => {
  if (!isMessageType(message)) {
    throw new Error('Invalid message format');
  }
  
  // Sanitize content
  message.content = sanitizeContent(message.content);
  return message;
};

const isMessageType = (message: unknown): message is MessageType => {
  if (typeof message !== 'object' || message === null) {
    return false;
  }
  
  const msg = message as any;
  return (
    typeof msg.type === 'string' &&
    ['status', 'event', 'image', 'token'].includes(msg.type)
  );
};

const sanitizeContent = (content: string): string => {
  // Implement content sanitization
  return content.replace(/<[^>]*>/g, '');
};
```

2. Connection Security
```typescript
const secureWebSocket = (url: string): WebSocket => {
  if (!url.startsWith('wss://') && !isLocalhost(url)) {
    throw new Error('WebSocket must use WSS in production');
  }
  
  const ws = new WebSocket(url);
  
  ws.onmessage = (event) => {
    try {
      const message = validateMessage(JSON.parse(event.data));
      handleMessage(message);
    } catch (error) {
      console.error('Invalid message received:', error);
    }
  };
  
  return ws;
};
```

## Data Protection

1. Sensitive Data Handling
```typescript
class DataProtection {
  private static readonly SENSITIVE_KEYS = ['password', 'token', 'key'];
  
  static sanitizeData(data: any): any {
    if (typeof data !== 'object') return data;
    
    const sanitized = { ...data };
    for (const key of Object.keys(sanitized)) {
      if (this.SENSITIVE_KEYS.includes(key.toLowerCase())) {
        sanitized[key] = '[REDACTED]';
      } else if (typeof sanitized[key] === 'object') {
        sanitized[key] = this.sanitizeData(sanitized[key]);
      }
    }
    return sanitized;
  }
  
  static logSafely(data: any): void {
    console.log(this.sanitizeData(data));
  }
}
```

2. Error Handling
```typescript
class SecureError extends Error {
  constructor(
    message: string,
    private readonly sensitive: boolean = false
  ) {
    super(message);
  }
  
  public toJSON(): any {
    return {
      message: this.sensitive ? 'Internal server error' : this.message,
      timestamp: new Date().toISOString()
    };
  }
}

const handleError = (error: unknown): void => {
  if (error instanceof SecureError) {
    // Log full error internally
    console.error(error);
    // Return safe error to user
    return error.toJSON();
  }
  
  // Generic error for unknown cases
  return new SecureError('Internal server error', true).toJSON();
};
```

## Performance Optimization

1. Resource Cleanup
```typescript
class ResourceManager {
  private resources: Set<{ cleanup: () => void }> = new Set();
  
  register(resource: { cleanup: () => void }): void {
    this.resources.add(resource);
  }
  
  cleanup(): void {
    for (const resource of this.resources) {
      try {
        resource.cleanup();
      } catch (error) {
        console.error('Cleanup error:', error);
      }
    }
    this.resources.clear();
  }
}

// Usage
const manager = new ResourceManager();

const setupWebSocket = (url: string): WebSocket => {
  const ws = new WebSocket(url);
  manager.register({
    cleanup: () => ws.close()
  });
  return ws;
};
```

2. Memory Management
```typescript
class MemoryManager {
  private static readonly MAX_CACHE_SIZE = 100;
  private cache: Map<string, { data: any; timestamp: number }>;
  
  constructor() {
    this.cache = new Map();
  }
  
  set(key: string, value: any): void {
    if (this.cache.size >= MemoryManager.MAX_CACHE_SIZE) {
      this.cleanup();
    }
    
    this.cache.set(key, {
      data: value,
      timestamp: Date.now()
    });
  }
  
  private cleanup(): void {
    const now = Date.now();
    const hour = 60 * 60 * 1000;
    
    for (const [key, value] of this.cache.entries()) {
      if (now - value.timestamp > hour) {
        this.cache.delete(key);
      }
    }
  }
}
```

## Code Quality

1. Type Safety
```typescript
type Nominal<T, Brand> = T & { readonly __brand: Brand };

type UserId = Nominal<string, 'userId'>;
type Token = Nominal<string, 'token'>;

const createUser = (id: string): UserId => id as UserId;
const createToken = (value: string): Token => value as Token;

function processUser(user: UserId): void {
  // Type-safe processing
}

// Error: Type 'string' is not assignable to type 'UserId'
processUser('123');

// OK: Proper type creation
const userId = createUser('123');
processUser(userId);
```

2. Error Prevention
```typescript
class Result<T, E> {
  private constructor(
    private value: T | null,
    private error: E | null
  ) {}
  
  static ok<T, E>(value: T): Result<T, E> {
    return new Result<T, E>(value, null);
  }
  
  static err<T, E>(error: E): Result<T, E> {
    return new Result<T, E>(null, error);
  }
  
  map<U>(fn: (value: T) => U): Result<U, E> {
    if (this.value === null) return Result.err(this.error!);
    return Result.ok(fn(this.value));
  }
  
  handle<U>(
    onOk: (value: T) => U,
    onErr: (error: E) => U
  ): U {
    if (this.value === null) return onErr(this.error!);
    return onOk(this.value);
  }
}

// Usage
const divide = (a: number, b: number): Result<number, string> => {
  if (b === 0) return Result.err('Division by zero');
  return Result.ok(a / b);
};

const result = divide(10, 2)
  .map(value => value * 2)
  .handle(
    value => `Result: ${value}`,
    error => `Error: ${error}`
  );
```

Remember:
- Validate all inputs
- Protect sensitive data
- Implement proper cleanup
- Use type safety features
- Handle errors gracefully
- Monitor resource usage