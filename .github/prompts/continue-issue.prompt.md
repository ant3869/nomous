---
description: 'Continue working on an unresolved issue using comprehensive MCP tool analysis, documentation research, and memory system context to identify and implement the complete solution.'
agent: agent
---

# Continue Working on Issue

The issue is not fully resolved yet. I need you to continue working on this problem using all available MCP tools and resources to systematically complete the implementation.

## Required MCP Tool Analysis

Before continuing, perform comprehensive analysis using these tools:

### 1. Memory System Context
Use #mcp_memory_read_graph to understand what we know about this issue and previous attempts. Search for related nodes with #mcp_memory_search_nodes to find context about what has been tried. Open specific nodes using #mcp_memory_open_nodes to review detailed information about components involved.

### 2. Documentation Research
Search Microsoft documentation with #mcp_microsoftdocs_microsoft_docs_search for relevant APIs, patterns, and best practices. Find official code samples using #mcp_microsoftdocs_microsoft_code_sample_search to see correct implementations. For third-party libraries, use #mcp_context7_resolve-library-id to identify library IDs, then #mcp_context7_get-library-docs to retrieve current documentation.

### 3. Current State Assessment
Review the codebase with #codebase to understand:
- What was attempted in previous iterations?
- What symptoms or errors are still occurring?
- What parts of the solution are working vs. not working?
- What assumptions were made that might be incorrect?
- Are there configuration issues we missed?
- Do we need to enable or initialize something?
- Is there a sequence or timing issue?

## Systematic Action Plan

1. **Gather Context**
   - Use #mcp_memory_read_graph to see what we learned from previous attempts
   - Search memory for similar issues or related components
   - Review error logs and console output

2. **Research Solutions**
   - Use #mcp_microsoftdocs_microsoft_docs_search for Microsoft/Azure APIs
   - Use #mcp_microsoftdocs_microsoft_code_sample_search for working examples
   - Use #mcp_context7_get-library-docs for third-party library documentation
   - Compare our implementation against official examples

3. **Analyze Implementation**
   - Review current code with fresh perspective
   - Check data flow from start to finish
   - Verify all initializations happen in correct order
   - Look for silent failures or unhandled errors
   - Validate configuration and settings

4. **Identify Gaps**
   - What's missing in the current approach?
   - Are there missing dependencies or imports?
   - Is error handling comprehensive?
   - Are there race conditions or timing issues?

5. **Implement Fix**
   - Based on documentation and examples
   - Add comprehensive logging
   - Include proper error handling
   - Test each component independently

6. **Update Memory**
   - Use #mcp_memory_add_observations to document findings
   - Use #mcp_memory_create_relations to link related concepts
   - Document what was wrong and how it was fixed

## Focus Areas for Nomous Project

### ML/AI Components
- Model loading and initialization sequence
- GPU memory management
- Inference pipeline completeness
- Error recovery mechanisms

### WebSocket Communication
- Connection state management
- Message serialization/deserialization
- Audio chunk streaming
- Protocol compliance

### Real-time Processing
- Microphone capture pipeline
- STT processing flow
- Audio encoding/decoding
- Buffer management

### Offline Capabilities
- Local model availability
- Resource accessibility
- Fallback mechanisms
- State persistence

## Expected Deliverables

1. **Clear Diagnosis**
   - Exact identification of what's still broken
   - Evidence from logs or testing
   - Root cause analysis

2. **Research Findings**
   - Relevant documentation URLs
   - Code sample references
   - Best practices identified

3. **Concrete Fixes**
   - Specific code changes with explanations
   - Configuration updates if needed
   - Proper error handling added

4. **Testing Instructions**
   - How to verify the fix works
   - What to look for in logs
   - Expected behavior description

5. **Memory Updates**
   - Observations added about what was learned
   - Relations created between components
   - Documentation for future reference

## Success Criteria

- [ ] Root cause identified with evidence
- [ ] Official documentation consulted
- [ ] Implementation matches best practices
- [ ] All MCP tool findings documented
- [ ] Code changes implemented and tested
- [ ] Comprehensive logging added
- [ ] Memory system updated
- [ ] Issue verified as completely resolved
