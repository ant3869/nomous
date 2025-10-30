---
description: 'WebSocket communication patterns and standards'
applyTo: '**/*.{ts,tsx,py}'
---

# WebSocket Communication Guidelines

## Message Protocol

- Use TypeScript interfaces for message types
- Implement proper serialization/deserialization
- Maintain protocol versioning
- Document message structures

## Connection Management

- Handle reconnection logic
- Implement proper connection lifecycle
- Manage connection state
- Handle cleanup properly

## Error Handling

- Define error message types
- Implement proper error propagation
- Handle disconnections gracefully
- Provide meaningful error feedback

## Performance

- Optimize message size
- Implement proper batching
- Handle backpressure
- Manage connection load

## Security

- Validate all messages
- Sanitize data properly
- Implement proper authentication
- Follow secure WebSocket practices

## Testing

- Test connection scenarios
- Validate message handling
- Test error conditions
- Verify protocol compliance