---
description: 'Deep investigation for persistent problems using comprehensive MCP tool analysis, comparing against official documentation, and questioning assumptions to find the root cause and correct solution.'
agent: agent
---

# Issue Not Fixed - Deep Investigation Required

The problem persists despite previous attempts. We need a thorough root cause investigation using all available MCP tools to identify what we're missing and implement the correct solution.

## Immediate MCP Tool Actions

### 1. Complete Memory Review
Use #mcp_memory_read_graph to review the complete knowledge graph and understand our full history. Search with #mcp_memory_search_nodes to find all nodes related to this issue. Use #mcp_memory_open_nodes to examine specific attempts, observations, and documented solutions. This tells us what we've already tried and what we learned.

### 2. Authoritative Documentation Deep Dive
Search official Microsoft documentation with #mcp_microsoftdocs_microsoft_docs_search to find authoritative guidance on the technologies involved. Use #mcp_microsoftdocs_microsoft_code_sample_search with appropriate language parameter to get working code examples. Compare our implementation line-by-line against these official samples.

### 3. Library Documentation Verification
For any third-party libraries, use #mcp_context7_resolve-library-id to get the correct library identifier. Then use #mcp_context7_get-library-docs with specific topics to get detailed documentation. Check for version-specific issues, breaking changes, or deprecated patterns we might be using.

### 4. Codebase Pattern Analysis
Use #codebase to search for similar functionality that works correctly in other parts of the project. Compare working implementations against the broken one to identify differences in approach, initialization order, or error handling.

## Deep Diagnostic Questions

### Architecture Level
1. Is the overall approach correct according to official documentation?
2. Are we using the right APIs and methods for this functionality?
3. Is there a simpler or more standard pattern we should follow?
4. Do example implementations follow a different architecture?

### Implementation Level
1. Are all required initializations happening?
2. Is the order of operations correct?
3. Are we waiting for async operations properly?
4. Are there race conditions or timing dependencies?
5. Is data flowing through the entire pipeline?
6. Are type conversions happening correctly?
7. Are we handling null/undefined cases?

### Configuration Level
1. Are all necessary features enabled in settings?
2. Are paths and file references correct and accessible?
3. Are dependencies properly installed with correct versions?
4. Are there environment-specific requirements?
5. Do we need special permissions or capabilities?

### Integration Level
1. Are all components properly connected?
2. Is the WebSocket communication working correctly?
3. Are events being fired and received?
4. Are message formats correct?
5. Is state being synchronized properly?

## Systematic Research Strategy

### Step 1: Project Memory Analysis
```
#mcp_memory_search_nodes query="[component/feature name]"
```
Review everything we've documented about this issue. What patterns emerge? What have we not tried yet?

### Step 2: Official Documentation Research
```
#mcp_microsoftdocs_microsoft_docs_search query="[API/technology name]"
#mcp_microsoftdocs_microsoft_code_sample_search query="[specific functionality]" language="[typescript/python/etc]"
```
Find the authoritative way to implement this. What do official examples show?

### Step 3: Library Documentation Verification
```
#mcp_context7_resolve-library-id libraryName="[package name]"
#mcp_context7_get-library-docs context7CompatibleLibraryID="[id]" topic="[feature area]"
```
Ensure we're using libraries correctly according to their current documentation.

### Step 4: Codebase Comparison
```
#codebase [search for working similar functionality]
```
How do working features handle similar scenarios? What can we learn from them?

## Root Cause Analysis Framework

### Go Back to Basics
1. **Exact Symptom**: What is the precise error, missing behavior, or incorrect output?
2. **Failure Point**: At what exact point does it fail? (Use logging to pinpoint)
3. **Assumptions Made**: List every assumption about how it should work
4. **Assumption Testing**: Could any assumption be wrong? How can we verify?
5. **Working Cases**: What scenarios DO work? What's different?

### Evidence Collection
- Console logs showing the failure
- Network tab showing WebSocket messages (or lack thereof)
- Browser DevTools errors
- Backend logs with timing and sequence
- Memory/CPU usage patterns

### Comparison Analysis
Create a side-by-side comparison:
- **Official Example** vs **Our Implementation**
- **Working Feature** vs **Broken Feature**  
- **Expected Behavior** vs **Actual Behavior**
- **Required Steps** vs **Steps We're Taking**

## New Approach Based on Research

After gathering all evidence and documentation:

1. **Identify Authoritative Pattern**
   - How do official docs say to implement this?
   - What initialization steps are required?
   - What order must operations occur in?
   - What error handling is recommended?

2. **List Specific Differences**
   - What are we doing differently from examples?
   - What are we missing from the official pattern?
   - What assumptions differ from documentation?
   - What edge cases aren't we handling?

3. **Implement Corrections**
   - Follow official patterns exactly
   - Add all recommended initialization
   - Implement proper error handling
   - Add comprehensive logging at each step
   - Handle edge cases explicitly

4. **Test Components Independently**
   - Verify each component works alone
   - Test integration step by step
   - Validate data at each stage
   - Confirm events fire as expected

## Documentation Requirements

### During Implementation
Add detailed logging that shows:
- What is being attempted
- What inputs are being used
- What outputs are produced
- What errors occur (with full details)
- Timing and sequence of operations

### After Fixing
Update memory with comprehensive findings:

```
#mcp_memory_add_observations entityName="[Component]" observations=[
  "Root cause was: [specific issue]",
  "Fixed by: [specific solution]",
  "Key learning: [important insight]",
  "Prevention: [how to avoid in future]"
]
```

Create relations to link concepts:
```
#mcp_memory_create_relations relations=[
  {from: "Error", to: "Solution", relationType: "resolved by"},
  {from: "Component", to: "Dependency", relationType: "requires"},
  {from: "Issue", to: "RootCause", relationType: "caused by"}
]
```

## Nomous-Specific Considerations

### ML/AI Pipeline
- Is the model loaded before inference?
- Is GPU memory allocated properly?
- Are tensors on the correct device?
- Is cleanup happening after inference?

### WebSocket Bridge
- Is the connection established before sending?
- Are message formats matching protocol?
- Is serialization/deserialization correct?
- Are errors propagating properly?

### Audio Processing
- Is the microphone permission granted?
- Is AudioContext initialized and resumed?
- Are audio chunks being encoded correctly?
- Is sample rate matching expectations?
- Is STT model loaded and configured?

### Real-time Processing
- Are buffers being managed correctly?
- Is backpressure handled?
- Are async operations awaited?
- Is state synchronized between components?

## Success Criteria

- [ ] Root cause identified with clear evidence
- [ ] Official documentation thoroughly consulted
- [ ] Implementation matches authoritative patterns
- [ ] All edge cases handled
- [ ] Comprehensive logging throughout
- [ ] Independent component testing passed
- [ ] Integration testing successful
- [ ] Memory updated with complete findings
- [ ] Prevention strategies documented
- [ ] Issue verified as permanently resolved
