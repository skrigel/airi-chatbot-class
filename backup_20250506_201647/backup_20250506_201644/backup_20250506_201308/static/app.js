// DOM elements
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-btn');
const resetButton = document.getElementById('reset-btn');

// Log that the script loaded
console.log("App.js loaded");

// Add event listeners when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded");
    
    // Explicitly add click handlers
    if (sendButton) {
        sendButton.onclick = function() {
            console.log("Send button clicked");
            sendMessage();
        };
        console.log("Added click handler to send button");
    } else {
        console.error("Send button not found!");
    }
    
    if (resetButton) {
        resetButton.onclick = function() {
            console.log("Reset button clicked");
            resetConversation();
        };
        console.log("Added click handler to reset button");
    } else {
        console.error("Reset button not found!");
    }
    
    // Add enter key handler to input
    if (messageInput) {
        messageInput.onkeypress = function(event) {
            if (event.key === 'Enter') {
                console.log("Enter key pressed");
                sendMessage();
            }
        };
        console.log("Added keypress handler to input");
    } else {
        console.error("Message input not found!");
    }
});

// Manually check elements and add listeners again (belt and suspenders)
window.onload = function() {
    console.log("Window loaded");
    
    // Log all important elements
    console.log({
        chatMessages: document.getElementById('chat-messages'),
        messageInput: document.getElementById('message-input'),
        sendButton: document.getElementById('send-btn'),
        resetButton: document.getElementById('reset-btn')
    });
    
    // Try again with direct onclick assignment
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) {
        sendBtn.onclick = sendMessage;
        console.log("Re-attached send button handler");
    }
    
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.onclick = resetConversation;
        console.log("Re-attached reset button handler");
    }
};

// Function to add a message to the chat
function addMessage(message, isUser = false) {
    console.log("Adding message:", message, "isUser:", isUser);
    
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

// Simple send message function - no streaming
function sendMessage() {
    console.log("sendMessage function called");
    
    const message = messageInput.value.trim();
    console.log("Message to send:", message);
    
    if (!message) {
        console.log("Empty message, not sending");
        return;
    }
    
    // Clear input and add user message
    messageInput.value = '';
    addMessage(message, true);
    
    // Add temporary bot message
    const tempMessage = addMessage("Processing your request...", false);
    
    // Make API request
    const url = window.location.origin + '/api/chat';
    console.log("Sending request to:", url);
    
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            conversation_id: 'default',
            stream: false
        })
    })
    .then(response => {
        console.log("Response status:", response.status);
        return response.json();
    })
    .then(data => {
        console.log("Response data:", data);
        // Remove temporary message
        tempMessage.remove();
        // Add bot response
        addMessage(data.response, false);
    })
    .catch(error => {
        console.error("Error:", error);
        // Remove temporary message
        tempMessage.remove();
        // Add error message
        addMessage("Sorry, there was an error processing your request. Check the console for details.", false);
    });
}

// Reset conversation
function resetConversation() {
    console.log("resetConversation function called");
    
    // Make API request
    const url = window.location.origin + '/api/reset';
    console.log("Sending reset request to:", url);
    
    fetch(url, {
        method: 'POST'
    })
    .then(response => {
        console.log("Reset response status:", response.status);
        return response.json();
    })
    .then(data => {
        console.log("Reset response data:", data);
        // Clear chat
        chatMessages.innerHTML = '';
        // Add welcome message
        addMessage("Hello! I'm the AI Risk Repository assistant. How can I help you navigate the repository today?", false);
    })
    .catch(error => {
        console.error("Reset error:", error);
        addMessage("Sorry, there was an error resetting the conversation. Check the console for details.", false);
    });
}

// Log that script completed execution
console.log("Script execution completed");