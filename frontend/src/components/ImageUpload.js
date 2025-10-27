/**
 * Image Upload Component
 * ======================
 * Drag-and-drop image upload with preview
 */

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileImage, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function ImageUpload({ onImageUpload, isProcessing }) {
  const [preview, setPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.dcm']
    },
    multiple: false,
    disabled: isProcessing
  });

  const handleUpload = () => {
    if (selectedFile) {
      onImageUpload(selectedFile);
    }
  };

  const handleClear = () => {
    setPreview(null);
    setSelectedFile(null);
  };

  return (
    <div className="card max-w-3xl mx-auto">
      <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-4 sm:mb-6 text-center">
        Upload X-ray Image
      </h2>

      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-xl p-6 sm:p-12 text-center cursor-pointer
              transition-all duration-300
              ${isDragActive
                ? 'border-primary bg-accent'
                : 'border-input hover:border-primary hover:bg-accent'
              }
              ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            <input {...getInputProps()} />
            <motion.div
              animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
              className="flex flex-col items-center"
            >
              <Upload className={`h-12 w-12 sm:h-16 sm:w-16 mb-3 sm:mb-4 ${isDragActive ? 'text-primary' : 'text-muted-foreground'}`} />
              <p className="text-base sm:text-lg font-semibold text-foreground mb-2">
                {isDragActive ? 'Drop the image here' : 'Drag & drop X-ray image'}
              </p>
              <p className="text-xs sm:text-sm text-muted-foreground mb-3 sm:mb-4">
                or click to select from your computer
              </p>
              <p className="text-xs text-muted-foreground/80">
                Supported formats: PNG, JPG, JPEG, DICOM
              </p>
            </motion.div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="space-y-4"
          >
            {/* Preview */}
            <div className="relative">
              <img
                src={preview}
                alt="Preview"
                className="w-full h-auto max-h-96 object-contain rounded-lg border-2"
              />
              <button
                onClick={handleClear}
                className="absolute top-2 right-2 bg-destructive/80 hover:bg-destructive text-destructive-foreground p-2 rounded-full shadow-lg transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* File Info */}
            <div className="flex items-center justify-between bg-muted rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <FileImage className="h-8 w-8 text-primary" />
                <div>
                  <p className="font-medium text-foreground">{selectedFile?.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(selectedFile?.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-3">
              <button
                onClick={handleUpload}
                disabled={isProcessing}
                className="btn-primary flex-1 w-full sm:w-auto"
              >
                {isProcessing ? 'Processing...' : 'Enhance Image'}
              </button>
              <button
                onClick={handleClear}
                disabled={isProcessing}
                className="btn-secondary w-full sm:w-auto"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Instructions */}
      <div className="mt-8 bg-accent border border-primary/20 rounded-lg p-4">
        <h3 className="font-semibold text-foreground mb-2">Tips for best results:</h3>
        <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
          <li>Use clear, high-resolution X-ray images</li>
          <li>Ensure the image is properly oriented</li>
          <li>Grayscale images work best</li>
          <li>The AI works best with chest X-rays</li>
        </ul>
      </div>
    </div>
  );
}

export default ImageUpload;
