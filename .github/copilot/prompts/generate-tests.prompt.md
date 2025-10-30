---
mode: 'agent'
description: 'Generate comprehensive test suites for WebSocket communication and ML model validation'
tools: ['edit', 'search']
---

# Test Suite Generator

Your goal is to create comprehensive test suites for WebSocket communication and ML model validation.

## Instructions

1. Analyze test requirements:
   - Component/module functionality
   - WebSocket communication patterns
   - ML model behavior
   - Performance criteria

2. Generate test structure:
   - Unit tests
   - Integration tests
   - Performance tests
   - Memory leak tests

3. Implement test cases:
   - Success scenarios
   - Error handling
   - Edge cases
   - Load testing

## Input Requirements

Provide:
- Component/module to test
- Expected behavior
- Performance requirements
- Error scenarios

## Success Criteria

- Complete test coverage
- Error case handling
- Performance validation
- Memory management checks

## Output Format

1. Test Structure:
   ```typescript
   describe('Module', () => {
     // Test organization
   });
   ```

2. Test Cases:
   ```typescript
   it('handles specific case', async () => {
     // Test implementation
   });
   ```

3. Mocks and Helpers:
   ```typescript
   const mockFunction = jest.fn();
   class MockWebSocket {
     // Mock implementation
   }
   ```