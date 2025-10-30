---
mode: 'agent'
description: 'Create and enhance React components with TypeScript and WebSocket integration'
tools: ['edit', 'search']
---

# React Component Generator

Your goal is to create or enhance React components with TypeScript support and WebSocket integration.

## Instructions

1. Analyze requirements:
   - Component purpose and features
   - Data model and props interface
   - WebSocket message types
   - Error handling needs

2. Generate component structure:
   - TypeScript interfaces
   - Functional component
   - Custom hooks
   - Error boundaries

3. Implement features:
   - State management
   - WebSocket communication
   - Error handling
   - Performance optimization

## Input Requirements

Provide:
- Component name and purpose
- Data model
- WebSocket message types
- Required features

## Success Criteria

- TypeScript type safety
- Proper error handling
- Optimized performance
- Clear documentation

## Output Format

1. Interfaces:
   ```typescript
   interface Props {
     // Props definition
   }
   
   type MessageType = {
     // Message type definition
   };
   ```

2. Component:
   ```typescript
   const Component = ({ prop1, prop2 }: Props) => {
     // Implementation
   };
   ```

3. Custom Hooks:
   ```typescript
   const useCustomHook = () => {
     // Hook implementation
   };
   ```

4. Tests:
   ```typescript
   describe('Component', () => {
     // Test cases
   });
   ```