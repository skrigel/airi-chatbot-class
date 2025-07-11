// API URL patch script for production deployment & Webflow integration
(function() {
  console.log("API URL patch applied - Webflow Integration Ready! 🚀");
  
  // Configure API base URL based on environment
  let API_BASE_URL;
  
  // Check for Webflow integration mode
  if (window.AIRI_CONFIG && window.AIRI_CONFIG.apiBaseUrl) {
    // Webflow embedded mode - use configured Railway backend
    API_BASE_URL = window.AIRI_CONFIG.apiBaseUrl;
    console.log('🌐 Webflow Integration Mode - Using Railway backend:', API_BASE_URL);
  } else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Local development - use backend on port 8090
    API_BASE_URL = 'http://localhost:8090';
    console.log('🛠️ Local Development Mode');
  } else {
    // Production Railway deployment - same origin
    API_BASE_URL = window.location.origin;
    console.log('🚂 Railway Production Mode');
  }
  
  console.log('API Base URL:', API_BASE_URL);
  
  // Override fetch to redirect API calls
  const originalFetch = window.fetch;
  window.fetch = function(url, options) {
    console.log('🔍 Intercepting fetch call to:', url);
    
    if (typeof url === 'string') {
      // Handle various localhost patterns
      if (url.includes('localhost:5000') || url.includes('localhost:8080') || url.includes('localhost:8090')) {
        const pathPart = url.split('/api/')[1] || url.split('/')[url.split('/').length - 1];
        const newUrl = `${API_BASE_URL}/api/${pathPart}`;
        console.log('🔄 Redirecting API call from', url, 'to', newUrl);
        url = newUrl;
      }
      // Handle relative API calls - this should catch /api/v1/stream or api/v1/stream
      else if (url.startsWith('/api/') || url.startsWith('api/') || url.includes('/api/')) {
        const apiPath = url.startsWith('/') ? url : `/${url}`;
        const newUrl = `${API_BASE_URL}${apiPath}`;
        console.log('🎯 Using API call:', newUrl);
        url = newUrl;
      }
    }
    console.log('📡 Final URL:', url);
    return originalFetch.call(this, url, options);
  };
})();