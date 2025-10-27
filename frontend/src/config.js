/**
 * API Configuration
 * Automatically detects environment and uses appropriate API URL
 */

// Get API URL from environment or use defaults
const getApiUrl = () => {
  // If REACT_APP_API_URL is set, use it
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }

  // In production (deployed), use relative /api path
  if (process.env.NODE_ENV === 'production') {
    return '/api';
  }

  // In development, use localhost
  return 'http://localhost:8000';
};

export const API_URL = getApiUrl();

export const API_ENDPOINTS = {
  enhance: `${API_URL}/enhance`,
  chatbot: `${API_URL}/chatbot`,
  health: `${API_URL}/health`,
};

console.log('API Configuration:', {
  mode: process.env.NODE_ENV,
  apiUrl: API_URL,
});
