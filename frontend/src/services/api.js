import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatService = {
  async askQuestion(question) {
    try {
      const response = await api.post('/chat', { question });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.detail || 'An error occurred while processing your question.');
      }
      throw new Error('Network error. Please ensure the backend is running.');
    }
  },

  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  async getModelInfo() {
    const response = await api.get('/model');
    return response.data;
  }
};

export default api;
