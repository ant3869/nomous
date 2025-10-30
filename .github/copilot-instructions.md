---
description: 'Repository-wide Copilot instructions for the Nomous project'
---

# Nomous Project Guidelines

## Project Overview

Nomous is an offline-first WebSocket bridge that connects a React dashboard to local AI capabilities including:
- Local LLM integration via llama.cpp (GGUF models)
- Speech recognition via Vosk
- Text-to-speech via Piper
- Computer vision via OpenCV

## Technology Stack

- **Frontend**: React 18+, TypeScript, Tailwind CSS, Vite
- **Backend**: Python with asyncio and WebSockets
- **ML/AI**: GGUF models, Vosk, Piper TTS
- **Build Tools**: Vite, PostCSS

## Core Development Principles

1. **Offline-First**: All functionality must work without internet connectivity
2. **Real-Time Processing**: Use WebSocket for streaming data between frontend and backend
3. **GPU Acceleration**: Utilize CUDA when available, fallback to CPU gracefully
4. **Type Safety**: Use TypeScript for frontend, maintain type hints in Python
5. **Component Architecture**: Follow atomic design principles for UI components
6. **Error Handling**: Comprehensive error handling with user-friendly fallbacks

## Code Organization

- Keep backend modules focused and single-responsibility
- Frontend components should be reusable and well-documented
- Use TypeScript interfaces for WebSocket message types
- Maintain clear separation between UI and business logic

## Testing Requirements

- Write unit tests for core functionality
- Include integration tests for WebSocket communication
- Test both GPU and CPU execution paths
- Validate error handling and fallback behaviors

## Performance Considerations

- Optimize WebSocket message size
- Use efficient data structures for ML operations
- Implement proper cleanup for GPU resources
- Monitor and optimize memory usage

## Security Guidelines

- Validate all user inputs
- Sanitize data before processing
- Implement proper error handling
- Follow secure WebSocket practices

## Documentation Standards

- Document all public APIs and interfaces
- Include usage examples in component documentation
- Maintain clear error message documentation
- Update configuration documentation as needed