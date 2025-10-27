/**
 * Footer Component - Mantra Health
 * ==================================
 */

import React from 'react';
import { Activity, Mail, Phone, MapPin, Github, Linkedin, Twitter } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center mb-4">
              <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-2 rounded-lg">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <span className="ml-2 text-xl font-bold">Mantra Health</span>
            </div>
            <p className="text-gray-400 mb-4">
              Revolutionizing medical imaging with artificial intelligence for better healthcare outcomes.
            </p>
            <div className="flex space-x-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                <Linkedin className="w-5 h-5" />
              </a>
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                <Twitter className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <a href="#features" className="text-gray-400 hover:text-white transition-colors">
                  Features
                </a>
              </li>
              <li>
                <a href="#how-it-works" className="text-gray-400 hover:text-white transition-colors">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#about" className="text-gray-400 hover:text-white transition-colors">
                  About Us
                </a>
              </li>
              <li>
                <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors text-left">
                  Pricing
                </button>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Resources</h3>
            <ul className="space-y-2">
              <li>
                <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors text-left">
                  Documentation
                </button>
              </li>
              <li>
                <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors text-left">
                  API Reference
                </button>
              </li>
              <li>
                <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors text-left">
                  Research Papers
                </button>
              </li>
              <li>
                <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors text-left">
                  Blog
                </button>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Contact Us</h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <Mail className="w-5 h-5 mr-2 text-purple-400 flex-shrink-0 mt-0.5" />
                <span className="text-gray-400">contact@mantrahealth.ai</span>
              </li>
              <li className="flex items-start">
                <Phone className="w-5 h-5 mr-2 text-purple-400 flex-shrink-0 mt-0.5" />
                <span className="text-gray-400">+1 (555) 123-4567</span>
              </li>
              <li className="flex items-start">
                <MapPin className="w-5 h-5 mr-2 text-purple-400 flex-shrink-0 mt-0.5" />
                <span className="text-gray-400">
                  123 AI Street, Tech Valley<br />
                  San Francisco, CA 94102
                </span>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-800 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm mb-4 md:mb-0">
              © {currentYear} Mantra Health. All rights reserved.
            </p>
            <div className="flex space-x-6 text-sm">
              <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors">
                Privacy Policy
              </button>
              <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors">
                Terms of Service
              </button>
              <button onClick={() => {}} className="text-gray-400 hover:text-white transition-colors">
                Cookie Policy
              </button>
            </div>
          </div>
          <p className="text-gray-500 text-xs mt-4 text-center md:text-left">
            For research and educational purposes only. Always consult healthcare professionals for medical decisions.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
