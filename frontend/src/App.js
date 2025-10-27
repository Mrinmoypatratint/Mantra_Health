/**
 * Main Application Component
 * ===========================
 * Mantra Health - X-ray Enhancement AI Frontend
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import LandingPage from './components/LandingPage';
import ImageUpload from './components/ImageUpload';
import ImageComparison from './components/ImageComparison';
import MetricsDisplay from './components/MetricsDisplay';
import ChatWidget from './components/ChatWidget';
import LoadingSpinner from './components/LoadingSpinner';
import Footer from './components/Footer';
import { Upload, Activity, Brain, MessageSquare, Home } from 'lucide-react';
import './App.css';
import { API_ENDPOINTS } from './config';

function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [originalImage, setOriginalImage] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [attentionMaps, setAttentionMaps] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [showChat, setShowChat] = useState(false);

  const handleImageUpload = async (file) => {
    setError(null);
    setIsProcessing(true);

    try {
      // Create FormData
      const formData = new FormData();
      formData.append('file', file);

      // Call backend API
      const response = await fetch(API_ENDPOINTS.enhance, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to enhance image');
      }

      const data = await response.json();

      // Set results
      setOriginalImage(URL.createObjectURL(file));
      setEnhancedImage(`data:image/png;base64,${data.enhanced_image}`);
      setMetrics(data.metrics);
      setAttentionMaps(data.attention_maps);

    } catch (err) {
      console.error('Error enhancing image:', err);
      setError('Failed to enhance image. Please make sure the backend server is running.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setEnhancedImage(null);
    setMetrics(null);
    setAttentionMaps(null);
    setError(null);
  };

  const handleGetStarted = () => {
    setShowLanding(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleBackToHome = () => {
    setShowLanding(true);
    handleReset();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Show Landing Page
  if (showLanding) {
    return (
      <>
        <LandingPage onGetStarted={handleGetStarted} />
        <Footer />
      </>
    );
  }

  // Show Main App
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm border-b sticky top-0 z-40 backdrop-blur-lg bg-white/95">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center space-x-3 sm:space-x-4">
              <button
                onClick={handleBackToHome}
                className="bg-gradient-to-r from-purple-600 to-pink-600 p-2 rounded-lg flex-shrink-0 hover:shadow-lg transition-shadow"
              >
                <Activity className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
              </button>
              <div>
                <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  Mantra Health
                </h1>
                <p className="text-xs sm:text-sm text-gray-600">
                  AI-Powered Medical Image Enhancement
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="hidden sm:flex items-center space-x-2 bg-purple-50 px-4 py-2 rounded-full">
                <Brain className="h-4 w-4 text-purple-600" />
                <span className="text-xs font-medium text-purple-600">
                  UNet + Attention + GAN
                </span>
              </div>
              <button
                onClick={handleBackToHome}
                className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-purple-600 transition-colors"
              >
                <Home className="h-4 w-4" />
                <span className="hidden sm:inline text-sm font-medium">Home</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mb-6 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-xl shadow-md"
            >
              <p className="font-medium">Error</p>
              <p className="text-sm">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Upload Section */}
        {!originalImage && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <ImageUpload onImageUpload={handleImageUpload} isProcessing={isProcessing} />
          </motion.div>
        )}

        {/* Processing Indicator */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center py-12"
          >
            <LoadingSpinner />
            <p className="mt-4 text-gray-600 font-medium">
              Enhancing X-ray image...
            </p>
            <p className="mt-1 text-sm text-gray-500">
              This may take a few seconds
            </p>
          </motion.div>
        )}

        {/* Results Section */}
        {originalImage && enhancedImage && !isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-6"
          >
            {/* Comparison Slider */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6">
                <h2 className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  Enhancement Results
                </h2>
                <button
                  onClick={handleReset}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-medium text-sm hover:shadow-lg transition-all duration-300 w-full sm:w-auto flex items-center justify-center space-x-2"
                >
                  <Upload className="w-4 h-4" />
                  <span>Upload New Image</span>
                </button>
              </div>
              <ImageComparison
                beforeImage={originalImage}
                afterImage={enhancedImage}
              />
            </div>

            {/* Metrics Display */}
            {metrics && (
              <MetricsDisplay metrics={metrics} />
            )}

            {/* Attention Maps */}
            {attentionMaps && (
              <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
                <h2 className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-6">
                  Attention Maps
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(attentionMaps).map(([key, value]) => (
                    <div key={key} className="space-y-2">
                      <img
                        src={`data:image/png;base64,${value}`}
                        alt={`Attention ${key}`}
                        className="w-full h-auto rounded-lg border border-gray-200"
                      />
                      <p className="text-sm text-center text-gray-600 font-medium">
                        {key.replace('_', ' ').toUpperCase()}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Features Section */}
        {!originalImage && !isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="mt-12 grid md:grid-cols-3 gap-6"
          >
            <FeatureCard
              icon={<Upload className="h-8 w-8 text-primary" />}
              title="Easy Upload"
              description="Simply drag and drop or select your X-ray image to get started"
            />
            <FeatureCard
              icon={<Brain className="h-8 w-8 text-primary" />}
              title="AI Enhancement"
              description="Advanced deep learning model with attention mechanism for superior results"
            />
            <FeatureCard
              icon={<Activity className="h-8 w-8 text-primary" />}
              title="Quality Metrics"
              description="PSNR, SSIM, and LPIPS metrics to quantify enhancement quality"
            />
          </motion.div>
        )}
      </main>

      {/* Chat Widget */}
      <ChatWidget isOpen={showChat} onToggle={() => setShowChat(!showChat)} />

      {/* Chat Toggle Button */}
      {!showChat && (
        <motion.button
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowChat(true)}
          className="fixed bottom-6 right-6 bg-primary hover:bg-primary/90 text-primary-foreground p-4 rounded-full shadow-lg z-50"
        >
          <MessageSquare className="h-6 w-6" />
        </motion.button>
      )}

      {/* Footer */}
      <Footer />
    </div>
  );
}

// Feature Card Component
function FeatureCard({ icon, title, description }) {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="card hover:shadow-xl transition-shadow duration-300"
    >
      <div className="flex flex-col items-center text-center">
        <div className="mb-4">{icon}</div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
        <p className="text-gray-600 text-sm">{description}</p>
      </div>
    </motion.div>
  );
}

export default App;
