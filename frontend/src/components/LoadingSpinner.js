/**
 * Loading Spinner Component
 * ==========================
 * Animated loading indicator
 */

import React from 'react';
import { motion } from 'framer-motion';

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center">
      <motion.div
        className="relative w-16 h-16"
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      >
        <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
        <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent"></div>
      </motion.div>
    </div>
  );
}

export default LoadingSpinner;
