---
description: 'Instructions for modern frontend development with React, TypeScript, and WebSockets'
---

# Frontend Development Instructions

## React Best Practices

1. Use Functional Components & Hooks
```typescript
const MyComponent = () => {
  const [state, setState] = useState<State>(initialState);
  
  useEffect(() => {
    // Side effects
    return () => {
      // Cleanup
    };
  }, [/* dependencies */]);
  
  return (
    // JSX
  );
};
```

2. Custom Hook Patterns
```typescript
const useWebSocket = (url: string) => {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    const socket = new WebSocket(url);
    
    socket.onopen = () => setIsConnected(true);
    socket.onclose = () => setIsConnected(false);
    
    setWs(socket);
    
    return () => {
      socket.close();
    };
  }, [url]);
  
  return { ws, isConnected };
};
```

3. Proper TypeScript Integration
```typescript
interface Props {
  data: DataType;
  onAction: (id: string) => void;
}

const Component = ({ data, onAction }: Props) => {
  // Implementation
};
```

## WebSocket Communication

1. Message Type Definitions
```typescript
type MessageType = 
  | { type: "status"; value: string; detail?: string }
  | { type: "event"; message: string }
  | { type: "image"; dataUrl: string }
  | { type: "token"; count: number };
```

2. WebSocket Manager
```typescript
class WebSocketManager {
  private socket: WebSocket;
  private messageHandlers: Map<string, (data: any) => void>;
  
  constructor(url: string) {
    this.socket = new WebSocket(url);
    this.messageHandlers = new Map();
    
    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const handler = this.messageHandlers.get(message.type);
      if (handler) handler(message);
    };
  }
  
  registerHandler(type: string, handler: (data: any) => void) {
    this.messageHandlers.set(type, handler);
  }
  
  send(message: MessageType) {
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }
}
```

3. Error Handling
```typescript
const connectWebSocket = async (url: string): Promise<WebSocket> => {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    
    socket.onopen = () => resolve(socket);
    socket.onerror = (error) => reject(error);
    
    setTimeout(() => {
      socket.close();
      reject(new Error("Connection timeout"));
    }, 5000);
  });
};
```

## State Management

1. Context + Reducer Pattern
```typescript
type State = {
  // State type
};

type Action = 
  | { type: "ACTION_1"; payload: any }
  | { type: "ACTION_2"; payload: any };

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case "ACTION_1":
      return { ...state, /* updates */ };
    case "ACTION_2":
      return { ...state, /* updates */ };
    default:
      return state;
  }
};

const AppContext = createContext<{
  state: State;
  dispatch: React.Dispatch<Action>;
} | undefined>(undefined);

const AppProvider = ({ children }: { children: React.ReactNode }) => {
  const [state, dispatch] = useReducer(reducer, initialState);
  
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};
```

## Performance Optimization

1. Memoization
```typescript
const MemoizedComponent = memo(({ data }: Props) => {
  return (
    // Expensive rendering
  );
}, (prevProps, nextProps) => {
  return prevProps.data.id === nextProps.data.id;
});

const memoizedCallback = useCallback(() => {
  // Callback implementation
}, [/* dependencies */]);

const memoizedValue = useMemo(() => {
  // Expensive computation
}, [/* dependencies */]);
```

2. Lazy Loading
```typescript
const LazyComponent = lazy(() => import('./HeavyComponent'));

const App = () => {
  return (
    <Suspense fallback={<Loading />}>
      <LazyComponent />
    </Suspense>
  );
};
```

3. Virtual Lists
```typescript
const VirtualList = ({ items }: { items: Item[] }) => {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 10 });
  
  return (
    <div className="virtual-list">
      {items
        .slice(visibleRange.start, visibleRange.end)
        .map(item => (
          <ListItem key={item.id} data={item} />
        ))}
    </div>
  );
};
```

## Error Boundaries

```typescript
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('Error:', error);
    console.error('Info:', info);
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback />;
    }
    
    return this.props.children;
  }
}
```

## Testing Patterns

1. Component Testing
```typescript
describe('Component', () => {
  it('renders correctly', () => {
    render(<Component {...props} />);
    expect(screen.getByText('expected text')).toBeInTheDocument();
  });
  
  it('handles user interaction', async () => {
    const onAction = jest.fn();
    render(<Component onAction={onAction} />);
    
    await userEvent.click(screen.getByRole('button'));
    expect(onAction).toHaveBeenCalled();
  });
});
```

2. Hook Testing
```typescript
describe('useWebSocket', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  
  it('connects to websocket', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost'));
    expect(result.current.isConnected).toBe(true);
  });
  
  it('handles disconnection', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost'));
    act(() => {
      // Simulate disconnection
      WebSocket.prototype.close.call(result.current.ws);
    });
    expect(result.current.isConnected).toBe(false);
  });
});
```

Remember:
- Use TypeScript for better type safety
- Implement proper error handling
- Add comprehensive testing
- Optimize for performance
- Follow React best practices