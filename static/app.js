// DOM elements
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-btn');
const resetButton = document.getElementById('reset-btn');

// API endpoint
const API_ENDPOINT = '/api';

// Conversation state
let conversationId = 'default';
let isWaitingForResponse = false;
let eventSource = null;

// Event listeners
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
resetButton.addEventListener('click', resetConversation);

// Function to add a message to the chat
function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    messageContent.textContent = message;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

// Function to add or update typing indicator
function showTypingIndicator(message = 'Bot is typing...') {
    let indicator = document.getElementById('typing-indicator');
    
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        chatMessages.appendChild(indicator);
    }
    
    indicator.textContent = message;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return indicator;
}

// Function to remove typing indicator
function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Function to handle streaming messages
function sendStreamingMessage(message) {
    if (message.length === 0 || isWaitingForResponse) {
        return;
    }
    
    // Clear input
    messageInput.value = '';
    
    // Add user message to chat
    addMessage(message, true);
    
    // Show typing indicator
    const indicator = showTypingIndicator('Initializing...');
    
    // Set waiting flag
    isWaitingForResponse = true;
    
    // Close any existing connection
    if (eventSource) {
        eventSource.close();
    }
    
    // Setup SSE connection
    eventSource = new EventSource(`${API_ENDPOINT}/chat?stream=true`);
    
    // Create POST request in background
    fetch(`${API_ENDPOINT}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            conversation_id: conversationId,
            stream: true
        })
    });
    
    // Handle incoming events
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.status === 'processing') {
            // Update typing indicator with status message
            indicator.textContent = data.message;
        } 
        else if (data.status === 'complete') {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add bot response to chat
            addMessage(data.response);
            
            // Update conversation ID if needed
            if (data.conversation_id) {
                conversationId = data.conversation_id;
            }
            
            // Close connection
            eventSource.close();
            eventSource = null;
            
            // Reset waiting flag
            isWaitingForResponse = false;
        }
        else if (data.status === 'error') {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add error message
            addMessage(data.message || 'Sorry, something went wrong. Please try again later.');
            
            // Close connection
            eventSource.close();
            eventSource = null;
            
            // Reset waiting flag
            isWaitingForResponse = false;
        }
    };
    
    eventSource.onerror = function() {
        // Hide typing indicator
        hideTypingIndicator();
        
        // Add error message
        addMessage('Connection lost. Please try again later.');
        
        // Close connection
        eventSource.close();
        eventSource = null;
        
        // Reset waiting flag
        isWaitingForResponse = false;
    };
}

// Function to send a message to the API (non-streaming fallback)
async function sendRegularMessage(message) {
    // Clear input
    messageInput.value = '';
    
    // Add user message to chat
    addMessage(message, true);
    
    // Show typing indicator
    showTypingIndicator();
    
    // Set waiting flag
    isWaitingForResponse = true;
    
    try {
        const response = await fetch(`${API_ENDPOINT}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId
            })
        });
        
        if (!response.ok) {
            throw new Error('Server error');
        }
        
        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Add bot response to chat
        addMessage(data.response);
        
        // Update conversation ID if needed
        if (data.conversation_id) {
            conversationId = data.conversation_id;
        }
    } catch (error) {
        console.error('Error:', error);
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Add error message
        addMessage('Sorry, something went wrong. Please try again later.');
    } finally {
        // Reset waiting flag
        isWaitingForResponse = false;
    }
}

// Main send message function - determines which approach to use
function sendMessage() {
    const message = messageInput.value.trim();
    
    if (message.length === 0 || isWaitingForResponse) {
        return;
    }
    
    // Check if browser supports EventSource
    if (window.EventSource) {
        sendStreamingMessage(message);
    } else {
        sendRegularMessage(message);
    }
}

// Function to reset the conversation
async function resetConversation() {
    if (isWaitingForResponse) {
        return;
    }
    
    try {
        const response = await fetch(`${API_ENDPOINT}/reset`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Server error');
        }
        
        // Clear chat messages
        chatMessages.innerHTML = '';
        
        // Add welcome message
        addMessage('Hello! I\'m the AI Risk Repository assistant. How can I help you navigate the repository today?');
        
        // Reset conversation ID
        conversationId = 'default';
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, failed to reset conversation. Please try again.');
    }
}