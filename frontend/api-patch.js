// API URL patch script
(function() {
  console.log("API URL patch applied");
  
  // Override fetch to redirect API calls
  const originalFetch = window.fetch;
  window.fetch = function(url, options) {
    if (typeof url === 'string' && url.includes('localhost:5000')) {
      const newUrl = url.replace('http://localhost:5000/', window.location.origin + '/');
      console.log('Redirecting API call from', url, 'to', newUrl);
      url = newUrl;
    }
    return originalFetch.call(this, url, options);
  };
})();
