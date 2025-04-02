import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navigation from './Navigation';
import './Upload.css';

const Upload = () => {
    const navigate = useNavigate();
    const [previewImage, setPreviewImage] = useState(null);
    const [cameraStream, setCameraStream] = useState(null);
    const [capturedImage, setCapturedImage] = useState(null);
    const [image, setImage] = useState(null);
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [user, setUser] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const userData = localStorage.getItem('user');
        if (!userData) {
            navigate('/');
            return;
        }
        setUser(JSON.parse(userData));
    }, [navigate]);

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Check file type
        if (!file.type.startsWith('image/')) {
            setError('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Check image dimensions
                if (img.width < 100 || img.height < 100) {
                    setError('Image is too small. Please upload a higher resolution image.');
                    return;
                }

                // Create a canvas to analyze the image
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                // Get image data
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;

                // Calculate average pixel intensity
                let totalIntensity = 0;
                let pixelCount = 0;
                for (let i = 0; i < data.length; i += 4) {
                    const r = data[i];
                    const g = data[i + 1];
                    const b = data[i + 2];
                    totalIntensity += (r + g + b) / 3;
                    pixelCount++;
                }
                const averageIntensity = totalIntensity / pixelCount;

                // Check if image characteristics match X-ray (grayscale, high contrast)
                const isGrayscale = checkGrayscale(data);
                const hasHighContrast = checkContrast(data);

                if (!isGrayscale || !hasHighContrast) {
                    setError('Please upload a valid X-ray image. The image should be grayscale and have good contrast.');
                    return;
                }

                // If all checks pass, proceed with the upload
                setPreviewImage(e.target.result);
                setImage(e.target.result);
                setError(null);
                sessionStorage.setItem('xrayImage', e.target.result);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    };

    // Helper function to check if image is grayscale
    const checkGrayscale = (data) => {
        let isGrayscale = true;
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            // Allow small differences due to compression
            if (Math.abs(r - g) > 5 || Math.abs(g - b) > 5 || Math.abs(r - b) > 5) {
                isGrayscale = false;
                break;
            }
        }
        return isGrayscale;
    };

    // Helper function to check image contrast
    const checkContrast = (data) => {
        let minIntensity = 255;
        let maxIntensity = 0;
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const intensity = (r + g + b) / 3;
            minIntensity = Math.min(minIntensity, intensity);
            maxIntensity = Math.max(maxIntensity, intensity);
        }
        // Calculate contrast ratio
        const contrastRatio = (maxIntensity - minIntensity) / 255;
        // X-ray images typically have high contrast
        return contrastRatio > 0.5;
    };

    const removeImage = (e) => {
        e.stopPropagation();
        setPreviewImage(null);
        sessionStorage.removeItem('xrayImage');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const openCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            setCameraStream(stream);
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.style.display = 'block';
            }
        } catch (error) {
            alert("Error accessing camera: " + error);
        }
    };

    const capturePhoto = () => {
        if (videoRef.current && canvasRef.current) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/png');
            setCapturedImage(imageData);
            sessionStorage.setItem('xrayImage', imageData);
            
            // Stop the camera stream
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                setCameraStream(null);
            }
            video.style.display = 'none';
        }
    };

    const retakePhoto = () => {
        setCapturedImage(null);
        sessionStorage.removeItem('xrayImage');
        openCamera();
    };

    const submitImage = () => {
        const imageToSubmit = previewImage || capturedImage;
        if (imageToSubmit) {
            navigate('/results');
        } else {
            alert('Please upload or capture an X-ray image first.');
        }
    };

    const handleLogout = () => {
        localStorage.removeItem('user');
        navigate('/');
    };

    if (!user) {
        return null;
    }

    return (
        <div className="upload-container">
            <Navigation />
            <div className="upload-content">
                <div className="page-title">
                    <h2>X-Ray Analysis</h2>
                    <p>Upload or capture an X-Ray image for analysis</p>
                </div>
                <div className="scan-options">
                    <div className="scan-option">
                        <h2>Upload X-Ray Image</h2>
                        <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleImageUpload}
                                accept="image/*"
                                style={{ display: 'none' }}
                            />
                            {!previewImage && (
                                <div className="upload-placeholder">
                                    <i className="fas fa-upload"></i>
                                    <p>Drop an Image <br /> or <br /> Click to Browse</p>
                                </div>
                            )}
                            {previewImage && (
                                <div className="preview-container">
                                    <img
                                        src={previewImage}
                                        alt="Preview"
                                        className="preview-image"
                                    />
                                    <button className="remove-button" onClick={removeImage}>
                                        Remove
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="scan-option">
                        <h2>Capture X-Ray Image</h2>
                        <div className="camera-box">
                            <video 
                                ref={videoRef} 
                                autoPlay 
                                style={{ display: 'none' }}
                            />
                            <canvas 
                                ref={canvasRef} 
                                style={{ display: 'none' }}
                            />
                            {!cameraStream && !capturedImage && (
                                <div className="camera-placeholder" onClick={openCamera}>
                                    <i className="fas fa-camera"></i>
                                    <p>Click to Open Camera</p>
                                </div>
                            )}
                            {capturedImage && (
                                <img src={capturedImage} alt="Captured X-ray" className="preview-image" />
                            )}
                        </div>
                        <div className="camera-buttons">
                            {cameraStream && !capturedImage && (
                                <button className="action-btn capture-btn" onClick={capturePhoto}>
                                    <i className="fas fa-camera"></i> Capture Photo
                                </button>
                            )}
                            {capturedImage && (
                                <button className="action-btn retake-btn" onClick={retakePhoto}>
                                    <i className="fas fa-redo"></i> Retake Photo
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                <button 
                    className="submit-btn" 
                    onClick={submitImage}
                    disabled={!previewImage && !capturedImage}
                >
                    Analyze Image
                </button>
            </div>
        </div>
    );
};

export default Upload; 