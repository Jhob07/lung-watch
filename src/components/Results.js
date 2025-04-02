import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Navigation from './Navigation';
import { analyzeImage } from '../services/modelService';
import './Results.css';

const Results = () => {
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [imageLoaded, setImageLoaded] = useState(false);
    const [isDrawing, setIsDrawing] = useState(false);
    const [currentColor, setCurrentColor] = useState('#ff0000');
    const [markers, setMarkers] = useState([]);
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [segmentationImage, setSegmentationImage] = useState(null);
    
    const imageRef = useRef(null);
    const canvasRef = useRef(null);
    const currentPathRef = useRef([]);

    useEffect(() => {
        // Check if user is logged in
        const userData = localStorage.getItem('user');
        if (!userData) {
            navigate('/login');
            return;
        }
        setUser(JSON.parse(userData));

        // Load X-ray image from session storage
        const xrayImage = sessionStorage.getItem('xrayImage');
        if (!xrayImage) {
            navigate('/upload');
            return;
        }
    }, [navigate]);

    const handleImageLoad = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const image = imageRef.current;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        setImageLoaded(true);
        // Automatically trigger analysis when image is loaded
        handleAnalyze();
    };

    useEffect(() => {
        if (imageLoaded) {
            setupCanvas(); // Call setupCanvas when the image is loaded
        }
    }, [imageLoaded]);

    const setupCanvas = () => {
        const canvas = canvasRef.current;
        const image = imageRef.current;
        if (canvas && image) {
            canvas.width = image.width;
            canvas.height = image.height;
        }
    };

    const startDrawing = (e) => {
        if (!imageLoaded) return;
        
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        setIsDrawing(true);
        currentPathRef.current = [{ x, y }];
    };

    const draw = (e) => {
        if (!isDrawing) return;
        
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        currentPathRef.current.push({ x, y });
        drawMarkers();
    };

    const stopDrawing = () => {
        if (!isDrawing) return;
        
        setIsDrawing(false);
        if (currentPathRef.current.length > 1) {
            setMarkers(prev => [...prev, {
                path: [...currentPathRef.current],
                color: currentColor
            }]);
        }
        currentPathRef.current = [];
        drawMarkers();
    };

    const drawMarkers = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw existing markers
        markers.forEach(marker => {
            ctx.beginPath();
            ctx.moveTo(marker.path[0].x, marker.path[0].y);
            marker.path.forEach(point => {
                ctx.lineTo(point.x, point.y);
            });
            ctx.strokeStyle = marker.color;
            ctx.lineWidth = 2;
            ctx.stroke();
        });
        
        // Draw current path
        if (currentPathRef.current.length > 1) {
            ctx.beginPath();
            ctx.moveTo(currentPathRef.current[0].x, currentPathRef.current[0].y);
            currentPathRef.current.forEach(point => {
                ctx.lineTo(point.x, point.y);
            });
            ctx.strokeStyle = currentColor;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    };

    const clearMarkers = () => {
        setMarkers([]);
        currentPathRef.current = [];
        drawMarkers();
    };

    const selectColor = (color) => {
        setCurrentColor(color);
    };

    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);
        
        try {
            // Get the original image from session storage
            const originalImage = sessionStorage.getItem('xrayImage');
            if (!originalImage) {
                throw new Error('No image data found');
            }

            // Create a new image element
            const img = new Image();
            img.src = originalImage;
            
            // Wait for the image to load
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });

            // Create a temporary canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = img.width;
            tempCanvas.height = img.height;
            const ctx = tempCanvas.getContext('2d');
            
            // Draw the original image
            ctx.drawImage(img, 0, 0);
            
            // Get the image data
            const imageData = tempCanvas.toDataURL('image/jpeg');
            console.log('Sending fresh image data for analysis...');
            
            const result = await analyzeImage(imageData);
            console.log('Analysis result received:', result);
            setAnalysis(result);
        } catch (err) {
            console.error('Analysis error:', err);
            setError('Failed to analyze image. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleAnalyzeAgain = () => {
        console.log("Navigating to upload page...");
        // Clear all states
        sessionStorage.removeItem('xrayImage');
        setAnalysis(null);
        setError(null);
        setLoading(false);
        setImageLoaded(false);
        setMarkers([]);
        currentPathRef.current = [];
        
        // Clear the canvas
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        
        // Clear any existing segmentation image
        setSegmentationImage(null);
        
        // Force a page reload to clear any cached states
        window.location.href = '/Upload';
    };

    if (!user) {
        return null;
    }

    return (
        <div className="results-container">
            <Navigation />
            <div className="results-content">
                <div className="image-section">
                    <div className="image-container">
                        <img
                            ref={imageRef}
                            src={sessionStorage.getItem('xrayImage')}
                            alt="X-Ray"
                            onLoad={handleImageLoad}
                        />
                        <canvas
                            ref={canvasRef}
                            className="drawing-canvas"
                            onMouseDown={startDrawing}
                            onMouseMove={draw}
                            onMouseUp={stopDrawing}
                            onMouseLeave={stopDrawing}
                        />
                    </div>
                    <div className="drawing-tools">
                        <div className="color-picker">
                            <button
                                className="color-button"
                                style={{ backgroundColor: '#ff0000' }}
                                onClick={() => selectColor('#ff0000')}
                            />
                            <button
                                className="color-button"
                                style={{ backgroundColor: '#00ff00' }}
                                onClick={() => selectColor('#00ff00')}
                            />
                            <button
                                className="color-button"
                                style={{ backgroundColor: '#0000ff' }}
                                onClick={() => selectColor('#0000ff')}
                            />
                        </div>
                        <button className="clear-button" onClick={clearMarkers}>
                            Clear Markers
                        </button>
                    </div>
                </div>
                
                <div className="analysis-section">
                    <h2>Analysis Results</h2>
                    {loading ? (
                        <div className="loading">Analyzing image...</div>
                    ) : error ? (
                        <div className="error">{error}</div>
                    ) : analysis ? (
                        <div className="analysis-results">
                            <div className={`result-card ${analysis.result.toLowerCase()}`}>
                                <h3>{analysis.result}</h3>
                                <p>Confidence: {analysis.confidence}</p>
                                {analysis.segmentation_path && (
                                    <img 
                                        src={`http://localhost:5000/${analysis.segmentation_path}`} 
                                        alt="Segmentation Result" 
                                        className="segmentation-result"
                                        onError={(e) => {
                                            console.error('Error loading segmentation image:', e);
                                            e.target.style.display = 'none';
                                        }}
                                    />
                                )}
                            </div>
                            <button 
                                className="analyze-again-button"
                                onClick={handleAnalyzeAgain}
                            >
                                Analyze Another Image
                            </button>
                        </div>
                    ) : (
                        <div className="no-analysis">
                            <p>Loading and analyzing image...</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Results; 