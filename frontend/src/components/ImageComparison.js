/**
 * Image Comparison Component
 * ===========================
 * Before/After slider for comparing original and enhanced images
 */

import React from 'react';
import {
  ReactCompareSlider,
  ReactCompareSliderImage
} from 'react-compare-slider';
import { motion } from 'framer-motion';

function ImageComparison({ beforeImage, afterImage }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Labels */}
      <div className="flex justify-between text-xs sm:text-sm font-medium text-gray-600">
        <span className="bg-red-100 text-red-800 px-2 sm:px-3 py-1 rounded-full">
          Original
        </span>
        <span className="bg-green-100 text-green-800 px-2 sm:px-3 py-1 rounded-full">
          Enhanced
        </span>
      </div>

      {/* Comparison Slider */}
      <div className="relative rounded-xl overflow-hidden border-2 border-gray-200 shadow-lg">
        <ReactCompareSlider
          itemOne={
            <ReactCompareSliderImage
              src={beforeImage}
              alt="Original X-ray"
              style={{
                objectFit: 'contain',
                backgroundColor: '#000'
              }}
            />
          }
          itemTwo={
            <ReactCompareSliderImage
              src={afterImage}
              alt="Enhanced X-ray"
              style={{
                objectFit: 'contain',
                backgroundColor: '#000'
              }}
            />
          }
          position={50}
          style={{
            height: '400px',
            width: '100%'
          }}
          className="sm:h-[600px]"
        />
      </div>

      {/* Instructions */}
      <p className="text-center text-xs sm:text-sm text-gray-500 italic">
        Drag the slider to compare original and enhanced images
      </p>
    </motion.div>
  );
}

export default ImageComparison;
