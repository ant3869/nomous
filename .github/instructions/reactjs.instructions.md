---
description: 'ReactJS development standards and best practices for Nomous frontend'
applyTo: 'src/frontend/**/*.{ts,tsx}'
---

# React Development Guidelines

## Component Structure

- Use functional components with hooks
- Keep components focused and single-responsibility
- Extract reusable logic into custom hooks
- Implement proper TypeScript interfaces

## State Management

- Use React hooks for local state
- Implement proper state updates with callbacks
- Handle WebSocket state efficiently
- Maintain clean state update patterns

## Performance Optimization

- Implement proper memoization with useMemo and useCallback
- Avoid unnecessary re-renders
- Optimize WebSocket message handling
- Use proper cleanup in useEffect hooks

## UI Components

- Follow shadcn/ui component patterns
- Maintain consistent styling with Tailwind
- Implement proper accessibility attributes
- Use proper TypeScript prop types

## Error Handling

- Implement proper error boundaries
- Handle WebSocket disconnections gracefully
- Show user-friendly error messages
- Provide fallback UI states

## Testing

- Write unit tests for components
- Test WebSocket integration
- Validate error handling
- Test accessibility compliance