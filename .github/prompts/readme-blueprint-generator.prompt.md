---
description: 'Intelligent README.md generation prompt that analyzes project documentation structure and creates comprehensive repository documentation. Scans .github/copilot directory files and copilot-instructions.md to extract project information, technology stack, architecture, development workflow, coding standards, and testing approaches, with a focus on offline-first ML/AI capabilities.'
mode: 'agent'
---

# README Generator Prompt

Generate a comprehensive README.md for this offline-first ML/AI bridge repository by analyzing the documentation files in the .github/copilot directory and the copilot-instructions.md file. Focus particularly on explaining the offline capabilities and ML/AI features.

1. Scan all files in the .github/copilot folder and related documentation:
   - ML and GPU acceleration guidelines
   - Python backend standards
   - ReactJS frontend standards
   - WebSocket communication patterns
   - Project architecture and data flows
   - Testing and deployment guidelines

2. Create a README.md with the following sections:

## Project Name and Description
- Extract the project name and primary purpose
- Emphasize the offline-first nature of the system
- Highlight the ML/AI capabilities and components
- List integrated technologies (llama.cpp, Vosk, Piper, OpenCV)

## Technology Stack
- List all primary technologies, languages, and frameworks
- Include specific versions and dependencies
- Emphasize ML/AI model requirements
- Document CUDA/GPU acceleration support
- List audio processing dependencies
- Include computer vision components

## Architecture
- Provide a high-level overview focusing on:
  - WebSocket bridge architecture
  - ML model integration patterns
  - Audio processing pipeline
  - Computer vision components
  - Frontend-backend communication
  - Offline data persistence

## Getting Started
- Installation instructions for all components
- GPU setup and configuration
- Model download and setup
- Development environment setup
- Local testing configuration
- Debug and profiling setup

## Project Structure
- Overview of repository organization
- Key directories and their purposes
- Model and data directories
- Configuration file locations
- Testing directory structure

## Key Features
- List all ML/AI capabilities
- Audio processing features
- Computer vision features
- Offline operation capabilities
- Real-time processing features
- WebSocket streaming features

## Development Guide
- Local development setup
- Testing procedures
- GPU profiling and optimization
- WebSocket debugging
- Model integration guidelines
- Performance optimization tips

## Contributing
- Contribution guidelines
- Code style requirements
- Testing requirements
- Documentation standards
- Performance benchmarking

## ML Model Management
- Model versioning and updates
- GGUF model configuration
- Voice model management
- Vision model setup
- Model optimization tips

## Performance Optimization
- GPU acceleration guidelines
- Memory management best practices
- WebSocket optimization tips
- Model inference optimization
- Real-time processing tips

Format the README with:
- Clear sections and subsections
- Code examples where appropriate
- Command-line instructions
- Performance notes and tips
- Links to detailed documentation
- GPU requirement badges

Keep focus on offline-first capabilities and real-time ML/AI processing while maintaining clear, concise documentation.