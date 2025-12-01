---
description: 'Systematic debugging of errors and problems using MCP tools for error research, code analysis, and documentation to diagnose root causes and implement robust solutions with proper error handling.'
agent: agent
---

# Debug Problems and Errors

There are errors or problems that need systematic diagnosis and resolution. Use comprehensive debugging with all available MCP tools to identify root causes and implement robust fixes.

## Immediate Error Assessment

### 1. Gather Complete Error Details
- What is the exact error message (copy verbatim)?
- Where does it occur (file, line number, function name)?
- When does it occur (startup, during operation, specific trigger)?
- What are the symptoms (crash, silent failure, incorrect behavior)?
- What is the stack trace or call chain?
- Are there multiple related errors?

### 2. Check Memory for Known Issues
```
#mcp_memory_search_nodes query="[error type or component name]"
#mcp_memory_read_graph
```
Have we encountered this error before? What did we learn? What solutions were documented?

### 3. Search Codebase for Context
```
#codebase [search for error location and related code]
```
Find where the error originates, trace the call stack, identify what's being passed to the failing code.

## Systematic Debugging with MCP Tools

### Step 1: Research the Error

#### Microsoft Documentation Search
```
#mcp_microsoftdocs_microsoft_docs_search query="[exact error message]"
#mcp_microsoftdocs_microsoft_docs_search query="[API name] error handling"
```
Find official guidance on this error, what causes it, and recommended solutions.

#### Code Sample Search
```
#mcp_microsoftdocs_microsoft_code_sample_search query="[functionality] error handling" language="[typescript/python/etc]"
```
Get working examples that show proper error handling for similar scenarios.

#### Library Documentation
```
#mcp_context7_resolve-library-id libraryName="[library causing error]"
#mcp_context7_get-library-docs context7CompatibleLibraryID="[id]" topic="troubleshooting"
#mcp_context7_get-library-docs context7CompatibleLibraryID="[id]" topic="error handling"
```
Get library-specific error documentation, common pitfalls, and solutions.

### Step 2: Analyze Code Context

Use #codebase to investigate:
- Where does the error originate?
- What is the complete call stack?
- What data is being passed to the failing code?
- Are there null/undefined values?
- Are data types correct?
- Are async operations handled properly?
- Is error handling present?

### Step 3: Compare with Working Examples

From MCP documentation tools:
- How do official examples handle this scenario?
- What error handling do they include?
- What initialization steps are required?
- What configuration is needed?
- What validation do they perform?

## Common Error Categories

### Configuration Errors
**Symptoms**: Feature not available, paths not found, settings ignored
**Investigation**:
- Verify all required settings are correctly set
- Check paths exist and are accessible
- Ensure required features are enabled in config
- Validate environment variables
- Check file permissions

### Initialization Errors  
**Symptoms**: Cannot access before initialization, object is null
**Investigation**:
- Is initialization happening in the correct order?
- Are all dependencies initialized before use?
- Are async initialization operations awaited?
- Are resources loaded before being accessed?
- Is timing/sequence correct?

### Data Flow Errors
**Symptoms**: Unexpected values, type errors, null reference
**Investigation**:
- Is data reaching the component?
- Is it in the expected format?
- Are type conversions correct?
- Are validation checks passing?
- Is null/undefined handled?
- Are defaults provided?

### Timing/Race Condition Errors
**Symptoms**: Intermittent failures, works sometimes
**Investigation**:
- Race conditions between async operations
- Components used before ready state
- Event handlers attached too late
- Resources accessed before loaded
- State updates out of order

### Permission/Access Errors
**Symptoms**: Permission denied, access forbidden
**Investigation**:
- Browser permissions for hardware (mic, camera)
- File system permissions
- CORS issues
- Authentication/authorization failures
- API key or credential problems

### Integration Errors
**Symptoms**: Communication failures, protocol mismatches
**Investigation**:
- WebSocket connection state
- Message format compatibility
- Protocol version mismatches
- Serialization/deserialization issues
- Network connectivity

## Debugging Tools Usage

### Memory Tools for Error Tracking
Document the investigation and solution:

```
#mcp_memory_add_observations entityName="[Component]" observations=[
  "Error: [exact error message]",
  "Occurs when: [trigger condition]",
  "Root cause: [what actually causes it]",
  "Solution: [how it was fixed]",
  "Prevention: [how to avoid in future]"
]
```

Create error solution entities:
```
#mcp_memory_create_entities entities=[
  {
    name: "[ErrorType]Solution",
    entityType: "KnownIssue",
    observations: [
      "Error pattern: [description]",
      "Common in: [scenarios]",
      "Fix approach: [solution]"
    ]
  }
]
```

Link errors to solutions:
```
#mcp_memory_create_relations relations=[
  {from: "[Component]", to: "[Error]", relationType: "can experience"},
  {from: "[Error]", to: "[Solution]", relationType: "resolved by"},
  {from: "[Error]", to: "[RootCause]", relationType: "caused by"}
]
```

### Documentation Tools for Solutions
Find best practices:
```
#mcp_microsoftdocs_microsoft_docs_search query="[error category] best practices"
#mcp_microsoftdocs_microsoft_code_sample_search query="[error scenario] handling"
```

Get library-specific patterns:
```
#mcp_context7_get-library-docs context7CompatibleLibraryID="[id]" topic="error handling"
```

## Solution Implementation

### 1. Add Comprehensive Logging
Before the error point:
- Log function entry with parameters
- Log conditional branch decisions
- Log data transformations
- Log async operation initiation

At the error point:
- Log inputs and their values
- Log current state
- Log what operation is being attempted

After the error point:
- Log success/failure explicitly
- Log return values
- Log state changes

### 2. Add Proper Error Handling
```typescript
try {
  // Log what we're about to do
  logger.info("Attempting operation X with input:", input);
  
  // Validate inputs
  if (!input || !input.requiredField) {
    throw new Error("Invalid input: missing requiredField");
  }
  
  // Perform operation
  const result = await riskyOperation(input);
  
  // Log success
  logger.info("Operation succeeded:", result);
  return result;
  
} catch (error) {
  // Log detailed error information
  logger.error("Operation failed:", {
    error: error.message,
    input: input,
    stack: error.stack
  });
  
  // Provide user-friendly message
  notifyUser("Operation failed: " + friendlyMessage(error));
  
  // Return safe default or throw
  return null; // or throw with context
}
```

### 3. Add Defensive Checks
- Verify prerequisites before operations
- Check state before state transitions  
- Validate data before processing
- Ensure resources exist before access
- Provide sensible defaults

### 4. Implement Graceful Degradation
- Fallback behaviors for errors
- User notifications when things fail
- Recovery mechanisms
- Retry logic where appropriate
- Safe default states

## Nomous-Specific Error Patterns

### ML/AI Errors
- Model not loaded: Check initialization sequence
- GPU out of memory: Implement cleanup after inference
- Inference timeout: Check batch size and model complexity
- Invalid input shape: Validate tensor dimensions

### WebSocket Errors
- Connection refused: Check server is running
- Message parse error: Validate JSON serialization
- Connection closed: Implement reconnection logic
- Protocol mismatch: Verify message format compliance

### Audio Processing Errors
- Permission denied: Request and handle microphone permission
- No audio device: Check device availability
- Sample rate mismatch: Implement resampling
- Buffer overflow: Implement backpressure handling

### Real-time Processing Errors
- Race conditions: Add proper async/await
- State desync: Implement state synchronization
- Memory leaks: Add proper cleanup
- Performance degradation: Profile and optimize

## Testing and Verification

### Test Error Path
1. Deliberately trigger the error condition
2. Verify fix prevents the error
3. Check error messages are clear and helpful
4. Ensure logging captures relevant information
5. Verify no new errors introduced

### Test Edge Cases
- What if called too early?
- What if data is missing or invalid?
- What if user denies permission?
- What if resource is unavailable?
- What if network is disconnected?

### Test Success Path
- Verify normal operation still works correctly
- Check performance isn't degraded
- Ensure logging is appropriate (not too verbose)
- Validate user experience is smooth
- Confirm all features still function

## Documentation After Fix

Update memory with complete findings:
```
#mcp_memory_add_observations entityName="[Component]" observations=[
  "Error was: [description]",
  "Root cause: [actual cause]",
  "Diagnosed using: [tools/methods]",
  "Fixed by: [solution]",
  "Prevention: [how to avoid]",
  "Testing: [how verified]"
]
```

Create prevention relations:
```
#mcp_memory_create_relations relations=[
  {from: "[ErrorPattern]", to: "[BestPractice]", relationType: "prevented by"},
  {from: "[Component]", to: "[Validation]", relationType: "requires"},
  {from: "[Error]", to: "[Logging]", relationType: "detected by"}
]
```

## Error Prevention Strategies

Based on findings, document:
1. What coding patterns led to this error?
2. What checks should be standard practice?
3. What validation should be automatic?
4. What documentation should be consulted first?
5. What testing should catch this type of error?
6. What monitoring should alert on similar issues?

Update project documentation with prevention guidelines.

## Debugging Checklist

- [ ] Error fully understood with exact reproduction steps
- [ ] Memory checked for previous occurrences
- [ ] Official documentation researched
- [ ] Code samples reviewed for comparison
- [ ] Library documentation consulted
- [ ] Root cause identified with evidence
- [ ] Fix implemented with proper error handling
- [ ] Comprehensive logging added throughout
- [ ] Tests verify fix works for error case
- [ ] Tests verify no regression in success case
- [ ] Edge cases handled appropriately
- [ ] User experience improved with clear messages
- [ ] Memory updated with findings and solution
- [ ] Prevention strategies documented
- [ ] Team/documentation updated with learnings
