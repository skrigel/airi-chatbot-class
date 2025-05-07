#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Path to the built frontend assets
const frontendDir = path.join(__dirname, 'github-frontend-build');
const assetsDir = path.join(frontendDir, 'assets');

// Get all JS files in the assets directory
console.log('Scanning directory:', assetsDir);
const files = fs.readdirSync(assetsDir).filter(file => file.endsWith('.js'));

if (files.length === 0) {
    console.error('No JS files found in assets directory');
    process.exit(1);
}

console.log(`Found ${files.length} JS files in assets directory`);

// The URL we want to replace and the new URL
const oldUrl = 'http://localhost:5000/';
const newUrl = window.location.origin + '/'; // Use the browser's origin

let fixedFiles = 0;

// Process each JS file
for (const file of files) {
    const filePath = path.join(assetsDir, file);
    console.log(`Checking file: ${filePath}`);
    
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        
        // Check if the file contains the old URL
        if (content.includes(oldUrl)) {
            console.log(`Found old URL in file: ${file}`);
            
            // Replace the URL
            const newContent = content.replace(
                'const API_URL = \'http://localhost:5000/\';', 
                'const API_URL = window.location.origin + \'/\';'
            );
            
            // Write the modified content back to the file
            fs.writeFileSync(filePath, newContent, 'utf8');
            console.log(`Fixed URL in file: ${file}`);
            fixedFiles++;
        }
    } catch (err) {
        console.error(`Error processing file ${file}:`, err);
    }
}

console.log(`Fixed ${fixedFiles} files`);

// Create a script to inject into index.html that will add our runtime config
const runtimeScript = `
// Patch the API URL at runtime
(function() {
  window.__AIRI_CONFIG = {
    apiUrl: window.location.origin + '/'
  };
  
  // Override fetch to rewrite URLs on the fly
  const originalFetch = window.fetch;
  window.fetch = function(url, options) {
    if (typeof url === 'string' && url.includes('localhost:5000')) {
      url = url.replace('http://localhost:5000/', window.location.origin + '/');
      console.log('Rewrote URL to:', url);
    }
    return originalFetch.call(this, url, options);
  };
})();
`;

// Create the script file
const scriptPath = path.join(assetsDir, 'api-config.js');
fs.writeFileSync(scriptPath, runtimeScript, 'utf8');
console.log(`Created runtime patch script at: ${scriptPath}`);

// Add the script to index.html
const indexPath = path.join(frontendDir, 'index.html');
let indexHtml = fs.readFileSync(indexPath, 'utf8');
const scriptTag = `<script src="/assets/api-config.js"></script>`;

// Add script before the closing head tag
indexHtml = indexHtml.replace('</head>', `${scriptTag}\n  </head>`);
fs.writeFileSync(indexPath, indexHtml, 'utf8');
console.log(`Added script tag to index.html`);

console.log('Frontend patch complete!');