---
description: 'Instructions for Python ML development including best practices for GPU optimization, model integration, and async patterns'
---

# Python ML Development Instructions

## GPU Optimization Guidelines

1. Ensure proper CUDA initialization
```python
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu_layers = 35  # Adjust based on GPU memory
else:
    device = torch.device("cpu")
    n_gpu_layers = 0
```

2. Implement proper resource cleanup
```python
try:
    # GPU operations
finally:
    torch.cuda.empty_cache()
```

3. Use efficient batch processing
```python
def process_batches(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch = batch.to(device)
        # Process batch
        torch.cuda.empty_cache()
```

## Model Integration Best Practices

1. Lazy loading for model initialization
```python
class ModelManager:
    def __init__(self):
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model
```

2. Implement model caching
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    return load_model()
```

3. Handle model versioning
```python
def load_model(version="latest"):
    model_path = get_model_path(version)
    if not model_path.exists():
        download_model(version)
    return Model.from_pretrained(model_path)
```

## Async Pattern Guidelines

1. Use async context managers
```python
class AsyncModelManager:
    async def __aenter__(self):
        self.model = await self.load_model()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.cleanup()
```

2. Implement proper cancellation
```python
import asyncio

async def process_with_timeout(data, timeout=30):
    try:
        async with asyncio.timeout(timeout):
            return await process_data(data)
    except asyncio.TimeoutError:
        # Handle timeout
```

3. Use efficient async queues
```python
async def process_queue():
    queue = asyncio.Queue()
    producers = [produce(queue) for _ in range(3)]
    consumers = [consume(queue) for _ in range(2)]
    await asyncio.gather(*producers, *consumers)
```

## Testing Guidelines

1. GPU test patterns
```python
def test_gpu_fallback():
    # Test graceful fallback to CPU
    with patch("torch.cuda.is_available", return_value=False):
        model = Model()
        assert model.device.type == "cpu"
```

2. Async test patterns
```python
@pytest.mark.asyncio
async def test_async_model():
    async with AsyncModelManager() as manager:
        result = await manager.process(data)
        assert result is not None
```

3. Memory leak detection
```python
def test_memory_cleanup():
    initial = torch.cuda.memory_allocated()
    process_data()
    torch.cuda.empty_cache()
    final = torch.cuda.memory_allocated()
    assert final <= initial
```

## Monitoring Guidelines

1. Track GPU metrics
```python
def log_gpu_stats():
    if torch.cuda.is_available():
        return {
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
            "max_memory_allocated": torch.cuda.max_memory_allocated()
        }
```

2. Performance profiling
```python
from torch.profiler import profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(input)
print(prof.key_averages().table())
```

## Error Handling

1. GPU-specific error handling
```python
try:
    result = model(input.to(device))
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        # Retry with smaller batch or on CPU
    raise
```

2. Model loading fallbacks
```python
def load_model_safe():
    try:
        return load_model_gpu()
    except Exception as e:
        logger.warning(f"GPU loading failed: {e}")
        return load_model_cpu()
```

Remember:
- Always implement proper GPU memory management
- Use async patterns for I/O operations
- Implement comprehensive error handling
- Add proper monitoring and logging
- Test both GPU and CPU paths