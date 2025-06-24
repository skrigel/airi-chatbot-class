// API URL patch script for production deployment
(function() {
  console.log("API URL patch applied");
  
  // Configure API base URL based on environment
  let API_BASE_URL;
  
  // Check if we're running in production (not localhost)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Local development - use current origin
    API_BASE_URL = window.location.origin;
  } else {
    // Production - use same origin (backend and frontend deployed together)
    // OR set a specific backend URL if deployed separately
    // Change this line to point to your Railway backend URL:
    API_BASE_URL = window.location.origin; // For same-origin deployment
    // API_BASE_URL = 'https://your-backend.up.railway.app'; // For separate deployment
  }
  
  console.log('API Base URL:', API_BASE_URL);
  
  // Override fetch to redirect API calls
  const originalFetch = window.fetch;
  window.fetch = function(url, options) {
    if (typeof url === 'string') {
      // Handle various localhost patterns
      if (url.includes('localhost:5000') || url.includes('localhost:8080') || url.includes('localhost:8090')) {
        const pathPart = url.split('/api/')[1] || url.split('/')[url.split('/').length - 1];
        const newUrl = `${API_BASE_URL}/api/${pathPart}`;
        console.log('Redirecting API call from', url, 'to', newUrl);
        url = newUrl;
      }
      // Handle relative API calls
      else if (url.startsWith('/api/')) {
        const newUrl = `${API_BASE_URL}${url}`;
        console.log('Using API call:', newUrl);
        url = newUrl;
      }
    }
    return originalFetch.call(this, url, options);
  };
})();
