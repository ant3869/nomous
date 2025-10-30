---
description: 'Instructions for comprehensive testing including WebSocket communication and ML model validation'
---

# Testing Best Practices

## WebSocket Testing

1. Mock WebSocket Server
```typescript
class MockWebSocket {
  private listeners: Map<string, Function[]> = new Map();
  
  constructor(url: string) {
    setTimeout(() => {
      this.emit('open');
    }, 0);
  }
  
  addEventListener(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)?.push(callback);
  }
  
  emit(event: string, data?: any) {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }
  
  send(data: string) {
    // Mock send
  }
  
  close() {
    this.emit('close');
  }
}

// Usage in tests
beforeEach(() => {
  global.WebSocket = MockWebSocket as any;
});
```

2. WebSocket Integration Tests
```typescript
describe('WebSocket Integration', () => {
  let mockServer: MockWebSocket;
  
  beforeEach(() => {
    mockServer = new MockWebSocket('ws://test');
  });
  
  it('handles connection', async () => {
    const { result } = renderHook(() => useWebSocket('ws://test'));
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });
  
  it('processes messages', async () => {
    const { result } = renderHook(() => useWebSocket('ws://test'));
    
    act(() => {
      mockServer.emit('message', { type: 'test', data: 'value' });
    });
    
    expect(result.current.messages).toContain('value');
  });
});
```

## ML Model Testing

1. Model Validation
```python
def test_model_output_shape():
    model = load_model()
    input_data = generate_test_input()
    output = model(input_data)
    
    assert output.shape == expected_shape
    assert output.dtype == expected_dtype

def test_model_gpu_fallback():
    with patch('torch.cuda.is_available', return_value=False):
        model = load_model()
        assert model.device.type == 'cpu'

def test_model_performance():
    model = load_model()
    input_data = generate_test_input()
    
    start_time = time.time()
    for _ in range(100):
        _ = model(input_data)
    elapsed = time.time() - start_time
    
    assert elapsed < max_allowed_time
```

2. Memory Management Tests
```python
def test_memory_cleanup():
    initial_memory = torch.cuda.memory_allocated()
    
    def process_data():
        data = generate_large_data()
        model(data)
        del data
    
    process_data()
    torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    assert final_memory <= initial_memory

@pytest.mark.asyncio
async def test_async_processing():
    async with AsyncModelManager() as manager:
        result = await manager.process(test_data)
        assert result is not None
```

## Component Testing

1. User Interaction Tests
```typescript
describe('Component Interaction', () => {
  it('handles user input', async () => {
    const onSubmit = jest.fn();
    render(<Component onSubmit={onSubmit} />);
    
    await userEvent.type(
      screen.getByRole('textbox'),
      'test input'
    );
    await userEvent.click(screen.getByRole('button'));
    
    expect(onSubmit).toHaveBeenCalledWith('test input');
  });
  
  it('shows loading state', async () => {
    render(<Component />);
    
    await userEvent.click(screen.getByRole('button'));
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });
});
```

2. Error Handling Tests
```typescript
describe('Error Handling', () => {
  it('shows error message', async () => {
    const error = new Error('Test error');
    jest.spyOn(console, 'error').mockImplementation(() => {});
    
    render(
      <ErrorBoundary>
        <ComponentThatThrows error={error} />
      </ErrorBoundary>
    );
    
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
  });
  
  it('recovers from errors', async () => {
    const { rerender } = render(
      <ErrorBoundary>
        <ComponentThatThrows error={new Error()} />
      </ErrorBoundary>
    );
    
    rerender(
      <ErrorBoundary>
        <ComponentThatThrows error={null} />
      </ErrorBoundary>
    );
    
    expect(screen.queryByText(/error/i)).not.toBeInTheDocument();
  });
});
```

## Performance Testing

1. Load Testing
```typescript
describe('Performance', () => {
  it('handles large data sets', async () => {
    const largeDataSet = generateLargeDataSet();
    const startTime = performance.now();
    
    render(<Component data={largeDataSet} />);
    
    const renderTime = performance.now() - startTime;
    expect(renderTime).toBeLessThan(maxRenderTime);
  });
  
  it('maintains FPS under load', async () => {
    const fps = await measureFPS(() => {
      render(<Component heavyOperation={true} />);
    });
    
    expect(fps).toBeGreaterThan(minFPS);
  });
});
```

2. Memory Leak Testing
```typescript
describe('Memory Management', () => {
  it('cleans up resources', async () => {
    const initialMemory = window.performance.memory?.usedJSHeapSize;
    
    const { unmount } = render(<Component />);
    unmount();
    
    const finalMemory = window.performance.memory?.usedJSHeapSize;
    expect(finalMemory).toBeLessThanOrEqual(initialMemory! * 1.1);
  });
});
```

Remember:
- Write comprehensive test suites
- Test both success and error cases
- Validate memory management
- Test performance under load
- Mock external dependencies
- Use proper cleanup in tests