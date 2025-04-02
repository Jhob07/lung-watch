import axios from 'axios';

const API_URL = 'http://localhost:5000';

export const analyzeImage = async (imageData) => {
    try {
        const response = await axios.post(`${API_URL}/analyze`, {
            image: imageData
        });
        return response.data;
    } catch (error) {
        console.error('Error analyzing image:', error);
        throw error;
    }
};

export const getAnalysisHistory = async () => {
    try {
        const response = await axios.get(`${API_URL}/history`);
        return response.data;
    } catch (error) {
        console.error('Error fetching analysis history:', error);
        throw error;
    }
}; 