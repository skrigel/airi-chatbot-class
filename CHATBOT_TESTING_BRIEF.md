# AI Risk Repository Chatbot: User Experience Testing Brief

## Background

The MIT AI Risk Repository project has developed an interactive chatbot interface to provide users with empirically-grounded, evidence-based information about AI risks. This chatbot serves as a conversational gateway to the comprehensive MIT AI Risk Repository, which contains thousands of documented AI risks, their impacts, and potential mitigations.

**Current Chatbot Capabilities:**
- Provides accurate, non-hallucinated responses based exclusively on AI Risk Repository data
- Employs advanced retrieval-augmented generation (RAG) with field-aware hybrid search
- Includes clickable citations linking directly to source documents
- Covers comprehensive AI risk domains including safety, privacy, bias, governance, and socioeconomic impacts

**Known Limitations:**
- Basic interface lacking user guidance and expectation management
- Response times of 5-10 seconds without user feedback during processing
- Limited context provided about the chatbot's purpose, scope, and capabilities
- Minimal formatting and visual hierarchy in response presentation
- Unclear link styling and interaction patterns

The chatbot represents a critical component of the AI Risk Repository's mission to make AI risk research accessible to policymakers, researchers, and practitioners. As we prepare for broader deployment, we need systematic user feedback to optimize the interface and user experience.

## Testing Objectives

For this testing phase, we aim to:

1. **Evaluate User Experience**: Assess the clarity, usability, and intuitiveness of the current chatbot interface
2. **Identify Friction Points**: Document specific moments where users experience confusion, frustration, or uncertainty
3. **Validate Information Architecture**: Determine whether users understand the chatbot's purpose, scope, and capabilities
4. **Optimize Response Presentation**: Gather feedback on formatting, citation display, and content organization
5. **Assess User Expectations**: Understand what users expect from an AI risk chatbot and how our implementation aligns with those expectations
6. **Prioritize Improvements**: Identify the most impactful modifications for enhancing user satisfaction and task completion

## Testing Protocol

### Pre-Testing Setup

**Access the Chatbot:**
- Navigate to [INSERT WEBSITE URL HERE]
- Ensure you have a stable internet connection
- Use a desktop or laptop browser (Chrome, Firefox, or Safari recommended)

**Recording Setup:**
- Install Veed screen recording software or use your preferred screen recording tool
- Begin recording before starting your interaction with the chatbot
- Include audio commentary describing your thoughts and reactions throughout the session

### Testing Session Structure

**Phase 1: Initial Impressions (3-5 minutes)**
1. Arrive at the chatbot interface without prior instruction
2. Spend 2-3 minutes exploring the interface before sending any messages
3. Verbally describe:
   - Your first impression of the interface
   - What you understand the chatbot's purpose to be
   - Any confusion or uncertainty about how to proceed
   - What additional information or guidance you would find helpful

**Phase 2: Natural Interaction (15-20 minutes)**
Engage with the chatbot using queries relevant to your interests or work. Consider testing different types of questions:

*Repository-Related Queries (recommended):*
- "What are the employment risks from AI automation?"
- "How does facial recognition bias affect marginalized communities?"
- "What privacy concerns arise from AI surveillance systems?"
- "What are the governance gaps in AI regulation?"

*General Knowledge Queries (to test boundaries):*
- "What's the capital of France?"
- "How do I bake cookies?"

*Edge Cases:*
- "Tell me about artificial flowers" (contains 'artificial' but not AI-related)
- "What machine should I buy for my gym?" (contains 'machine' but not AI-related)

**During interaction, please:**
- Think aloud as you formulate questions
- Comment on response times and any periods of uncertainty
- React to the content, formatting, and presentation of responses
- Attempt to click on citations and describe the experience
- Note any moments of confusion or delight

**Phase 3: Focused Evaluation (5-10 minutes)**
After natural interaction, please specifically evaluate:

1. **Response Quality**: Are the responses helpful, accurate, and appropriately scoped?
2. **Citation System**: Are the clickable links useful? Do they provide adequate source verification?
3. **User Guidance**: Do you feel adequately informed about what the chatbot can and cannot do?
4. **Visual Design**: How would you rate the interface's aesthetics and professionalism?
5. **Performance**: How do the response times affect your experience?

## Documentation Requirements

### Screen Recording
- **Duration**: 25-35 minutes total
- **Format**: MP4 or similar standard video format
- **Audio**: Include verbal commentary throughout your session
- **File Naming**: `[YourName]_ChatbotTesting_[Date].mp4`

### Post-Session Feedback Form

Please complete the following structured feedback within 24 hours of your testing session:

**Section A: Overall Experience**
1. On a scale of 1-10, how would you rate your overall experience with the chatbot?
2. What was your primary goal or expectation when first encountering the interface?
3. How well did the chatbot meet your expectations? What gaps remained?

**Section B: Usability Assessment**
1. **Interface Clarity**: How easily could you understand how to interact with the chatbot?
2. **Response Time**: How did the 5-10 second response delays affect your experience?
3. **Content Presentation**: How well-formatted and readable were the responses?
4. **Citation System**: How useful were the clickable source links?

**Section C: Content Evaluation**
1. **Information Quality**: How accurate and helpful were the responses to your AI risk questions?
2. **Scope Understanding**: How well did you understand what topics the chatbot could address?
3. **Boundary Management**: How appropriately did the chatbot handle non-AI risk questions?

**Section D: Improvement Priorities**
1. **Critical Issues**: What problems most hindered your ability to use the chatbot effectively?
2. **Quick Wins**: What small changes would significantly improve your experience?
3. **Feature Requests**: What additional capabilities would enhance the chatbot's utility for your work?

**Section E: Comparative Context**
1. **Similar Tools**: How does this chatbot compare to other AI or research tools you've used?
2. **User Background**: Briefly describe your familiarity with AI risk topics and your professional context
3. **Use Cases**: In what scenarios would you be most likely to use this chatbot?

## Technical Considerations

**Data Collection Transparency**: Please note that the backend system logs all user inputs for analysis and improvement purposes. Your interactions will be used to enhance the chatbot's functionality and user experience. Personal information should not be included in your test queries.

**System Architecture**: The chatbot employs advanced retrieval techniques including:
- Field-aware hybrid search combining vector similarity and keyword matching
- Multi-model fallback for enhanced reliability
- Semantic intent classification to optimize response relevance
- Real-time citation consistency validation

**Performance Context**: Current response times of 5-10 seconds reflect the system's comprehensive search and validation processes. We are exploring options for progress indicators and interim feedback during processing.

## Timeline and Next Steps

**Testing Schedule**:
- **Week 1**: Internal testing with Future Tech team members (3-4 testers)
- **Week 2**: External testing with domain experts (1-2 additional testers)
- **Week 3**: Analysis of feedback and implementation planning
- **Week 4**: Implementation of high-priority improvements

**Feedback Integration**:
- Weekly team meetings every Friday at 3pm EST to review findings
- Iterative improvement cycles based on prioritized feedback
- Follow-up testing sessions after significant modifications

**Deliverables from Testing**:
- Comprehensive feedback analysis report
- Prioritized improvement roadmap
- UI/UX enhancement specifications
- Technical optimization recommendations

## Long-term Vision

This testing phase supports our broader vision of creating the premier resource for evidence-based AI risk information. Key strategic objectives include:

**Differentiation**: Positioning the chatbot as uniquely empirically-grounded compared to other AI risk resources
**Personalization**: Potential future development of user-specific content curation based on professional background and research interests  
**Integration**: Possible collaboration with other AI risk initiatives (e.g., AI Risk Nexus, Vector Institute) for comprehensive coverage
**Scalability**: Ensuring the interface can support growing user bases across policy, research, and industry communities

## Contact Information

**For Technical Issues**: [INSERT TECHNICAL CONTACT]
**For Content Questions**: [INSERT CONTENT CONTACT]  
**For General Feedback**: [INSERT PROJECT LEAD CONTACT]

**Project Team**:
- Peter Slattery (Project Lead)
- David Turturean (Technical Implementation)
- Sasha [Last Name] (UI/UX Development)

---

*This brief was developed based on the June 27, 2025 project planning meeting and reflects current priorities for chatbot optimization. Please reach out with any questions about the testing protocol or feedback requirements.*