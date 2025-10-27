/**
 * Metrics Display Component
 * ==========================
 * Shows image quality metrics (PSNR, SSIM, LPIPS)
 */

import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Target, Zap } from 'lucide-react';

function MetricsDisplay({ metrics }) {
  const getQualityLevel = (psnr) => {
    if (psnr > 30) return { level: 'Excellent', color: 'green' };
    if (psnr > 25) return { level: 'Good', color: 'blue' };
    if (psnr > 20) return { level: 'Fair', color: 'yellow' };
    return { level: 'Poor', color: 'red' };
  };

  const getSSIMLevel = (ssim) => {
    if (ssim > 0.9) return { level: 'Excellent', color: 'green' };
    if (ssim > 0.8) return { level: 'Good', color: 'blue' };
    if (ssim > 0.7) return { level: 'Fair', color: 'yellow' };
    return { level: 'Poor', color: 'red' };
  };

  const psnrQuality = getQualityLevel(metrics.psnr);
  const ssimQuality = getSSIMLevel(metrics.ssim);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="card"
    >
      <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-6">
        Enhancement Quality Metrics
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        {/* PSNR Metric */}
        <MetricCard
          icon={<TrendingUp className="h-8 w-8" />}
          title="PSNR"
          value={`${metrics.psnr.toFixed(2)} dB`}
          description="Peak Signal-to-Noise Ratio"
          quality={psnrQuality.level}
          color={psnrQuality.color}
          info="Higher values indicate better image quality and less noise"
        />

        {/* SSIM Metric */}
        <MetricCard
          icon={<Target className="h-8 w-8" />}
          title="SSIM"
          value={metrics.ssim.toFixed(4)}
          description="Structural Similarity Index"
          quality={ssimQuality.level}
          color={ssimQuality.color}
          info="Values closer to 1 indicate better structural preservation"
        />

        {/* LPIPS Metric (if available) */}
        {metrics.lpips !== undefined && (
          <MetricCard
            icon={<Zap className="h-8 w-8" />}
            title="LPIPS"
            value={metrics.lpips.toFixed(4)}
            description="Perceptual Similarity"
            quality="N/A"
            color="gray"
            info="Lower values indicate better perceptual quality"
          />
        )}
      </div>

      {/* Overall Assessment */}
      <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Overall Assessment</h3>
        <p className="text-sm text-blue-800">
          {metrics.psnr > 25 && metrics.ssim > 0.8
            ? 'Excellent enhancement! The image quality has been significantly improved with minimal structural distortion.'
            : metrics.psnr > 20 && metrics.ssim > 0.7
            ? 'Good enhancement! The image quality has improved while maintaining important structural details.'
            : 'Enhancement applied. Some fine details may require careful examination.'}
        </p>
      </div>
    </motion.div>
  );
}

function MetricCard({ icon, title, value, description, quality, color, info }) {
  const colorClasses = {
    green: 'text-green-600 bg-green-50 border-green-200',
    blue: 'text-blue-600 bg-blue-50 border-blue-200',
    yellow: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    red: 'text-red-600 bg-red-50 border-red-200',
    gray: 'text-gray-600 bg-gray-50 border-gray-200',
  };

  const iconColorClasses = {
    green: 'text-green-600',
    blue: 'text-blue-600',
    yellow: 'text-yellow-600',
    red: 'text-red-600',
    gray: 'text-gray-600',
  };

  const badgeColorClasses = {
    green: 'bg-green-100 text-green-800',
    blue: 'bg-blue-100 text-blue-800',
    yellow: 'bg-yellow-100 text-yellow-800',
    red: 'bg-red-100 text-red-800',
    gray: 'bg-gray-100 text-gray-800',
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className={`metric-card ${colorClasses[color]} transition-transform duration-200`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className={iconColorClasses[color]}>{icon}</div>
        {quality !== 'N/A' && (
          <span className={`text-xs font-semibold px-2 py-1 rounded-full ${badgeColorClasses[color]}`}>
            {quality}
          </span>
        )}
      </div>

      <h3 className="text-lg font-bold text-gray-900 mb-1">{title}</h3>
      <p className="text-3xl font-extrabold text-gray-900 mb-2">{value}</p>
      <p className="text-sm text-gray-600 mb-2">{description}</p>

      {info && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500 italic">{info}</p>
        </div>
      )}
    </motion.div>
  );
}

export default MetricsDisplay;
