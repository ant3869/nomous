---
description: 'Python backend development standards for Nomous'
applyTo: 'src/backend/**/*.py'
---

# Python Backend Guidelines

## Code Organization

- Use asyncio for async operations
- Implement proper type hints
- Follow PEP 8 standards
- Maintain clean module structure

## WebSocket Implementation

- Use asyncio websockets
- Implement proper message serialization
- Handle connection lifecycle properly
- Maintain clean protocol structure

## ML Model Integration

- Proper GGUF model loading
- Efficient GPU resource management
- Clean model lifecycle handling
- Proper error handling for ML operations

## Performance

- Optimize WebSocket message handling
- Implement proper async patterns
- Manage GPU memory efficiently
- Use appropriate data structures

## Error Handling

- Implement proper exception handling
- Log errors appropriately
- Send meaningful error messages
- Handle resource cleanup

## Testing

- Write async-aware unit tests
- Test ML model integration
- Validate WebSocket handlers
- Test GPU and CPU paths