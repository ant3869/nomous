---
description: 'ML/GPU acceleration guidelines and best practices'
applyTo: 'src/backend/**/*.py'
---

# ML and GPU Guidelines

## Model Management

- Proper GGUF model loading
- Efficient model lifecycle
- Handle model versioning
- Implement proper cleanup

## GPU Acceleration

- Use CUDA when available
- Implement CPU fallback
- Manage GPU memory properly
- Monitor GPU utilization

## Resource Management

- Implement proper cleanup
- Monitor memory usage
- Handle OOM conditions
- Manage concurrent operations

## Performance Optimization

- Optimize batch processing
- Implement proper caching
- Use efficient data structures
- Monitor inference times

## Error Handling

- Handle GPU errors gracefully
- Provide fallback paths
- Log hardware issues
- Monitor resource usage

## Testing

- Test GPU and CPU paths
- Validate resource cleanup
- Monitor performance metrics
- Test error conditions