/**
 * Mantra Health - Professional Landing Page
 * ==========================================
 */

import React, { useState, useEffect } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import ChatWidget from './ChatWidget';
import {
  Activity,
  Brain,
  Upload,
  Zap,
  Shield,
  TrendingUp,
  Users,
  Award,
  CheckCircle,
  Play,
  ChevronRight,
  Sparkles,
  Heart,
  Target,
  Eye,
  MessageSquare,
  Bot,
  Lightbulb,
  Clock
} from 'lucide-react';

const LandingPage = ({ onGetStarted }) => {
  const [activeFeature, setActiveFeature] = useState(0);
  const [showChat, setShowChat] = useState(false);
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.2], [1, 0.95]);

  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: "AI-Powered Enhancement",
      description: "Advanced deep learning models (UNet + Attention + GAN) enhance X-ray clarity and detail",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Real-Time Processing",
      description: "Get enhanced X-ray results in seconds with our optimized inference pipeline",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Medical-Grade Quality",
      description: "PSNR, SSIM, and LPIPS metrics ensure clinical-quality enhancement",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: "Attention Visualization",
      description: "See what the AI focuses on with interactive attention maps",
      color: "from-orange-500 to-red-500"
    },
    {
      icon: <Eye className="w-8 h-8" />,
      title: "Before/After Comparison",
      description: "Interactive slider to compare original and enhanced images side-by-side",
      color: "from-indigo-500 to-purple-500"
    },
    {
      icon: <Award className="w-8 h-8" />,
      title: "Research-Backed",
      description: "Built on cutting-edge research in medical image enhancement",
      color: "from-pink-500 to-rose-500"
    }
  ];

  const stats = [
    { value: "99.8%", label: "Accuracy Rate", icon: <Target className="w-6 h-6" /> },
    { value: "10K+", label: "Images Processed", icon: <Upload className="w-6 h-6" /> },
    { value: "<3s", label: "Processing Time", icon: <Zap className="w-6 h-6" /> },
    { value: "100%", label: "Secure & Private", icon: <Shield className="w-6 h-6" /> }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % features.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [features.length]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      {/* Hero Section */}
      <motion.section
        style={{ opacity, scale }}
        className="relative min-h-screen flex items-center justify-center overflow-hidden"
      >
        {/* Animated Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
          <div className="absolute top-40 left-40 w-80 h-80 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex items-center justify-center mb-8"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full blur-lg opacity-50"></div>
              <div className="relative bg-gradient-to-r from-purple-600 to-pink-600 p-4 rounded-full">
                <Activity className="w-12 h-12 text-white" />
              </div>
            </div>
            <h1 className="ml-4 text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent">
              Mantra Health
            </h1>
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-3xl md:text-5xl lg:text-6xl font-bold text-gray-900 mb-6"
          >
            Revolutionizing Medical Imaging with
            <span className="block mt-2 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              Artificial Intelligence
            </span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="text-xl md:text-2xl text-gray-600 mb-12 max-w-3xl mx-auto"
          >
            Transform low-quality X-rays into crystal-clear diagnostic images using cutting-edge deep learning technology
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <button
              onClick={onGetStarted}
              className="group relative px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-semibold text-lg overflow-hidden transition-all duration-300 hover:shadow-2xl hover:scale-105"
            >
              <span className="relative z-10 flex items-center justify-center">
                Get Started Free
                <ChevronRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
              <div className="absolute inset-0 bg-gradient-to-r from-pink-600 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            </button>
            <button
              onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
              className="px-8 py-4 bg-white text-gray-900 rounded-full font-semibold text-lg border-2 border-gray-300 hover:border-purple-600 transition-all duration-300 hover:shadow-lg flex items-center justify-center"
            >
              <Play className="mr-2 w-5 h-5" />
              Watch Demo
            </button>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8"
          >
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                className="bg-white/80 backdrop-blur-lg rounded-2xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-shadow"
              >
                <div className="flex justify-center mb-3 text-purple-600">
                  {stat.icon}
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-1">{stat.value}</div>
                <div className="text-sm text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1.5 }}
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        >
          <div className="animate-bounce">
            <ChevronRight className="w-6 h-6 text-gray-400 transform rotate-90" />
          </div>
        </motion.div>
      </motion.section>

      {/* Features Section */}
      <section className="py-20 bg-white" id="features">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center px-4 py-2 bg-purple-100 text-purple-600 rounded-full mb-4">
              <Sparkles className="w-4 h-4 mr-2" />
              <span className="text-sm font-semibold">Features</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Powerful Features for Healthcare
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Built with the latest AI technology to deliver exceptional results
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                onMouseEnter={() => setActiveFeature(index)}
                className={`group relative bg-gradient-to-br ${activeFeature === index ? feature.color : 'from-gray-50 to-gray-100'} rounded-2xl p-8 transition-all duration-500 hover:shadow-2xl cursor-pointer`}
              >
                <div className={`${activeFeature === index ? 'text-white' : 'text-gray-900'} transition-colors duration-500`}>
                  <div className="mb-4">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                  <p className={`${activeFeature === index ? 'text-white/90' : 'text-gray-600'} transition-colors duration-500`}>
                    {feature.description}
                  </p>
                </div>
                {activeFeature === index && (
                  <motion.div
                    layoutId="activeFeature"
                    className="absolute inset-0 border-2 border-white rounded-2xl"
                    initial={false}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-gradient-to-b from-gray-50 to-white" id="how-it-works">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-600 rounded-full mb-4">
              <Play className="w-4 h-4 mr-2" />
              <span className="text-sm font-semibold">How It Works</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Simple. Fast. Effective.
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Transform your X-ray images in three easy steps
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 mb-16">
            {[
              {
                step: "01",
                title: "Upload Your X-ray",
                description: "Drag and drop or select your X-ray image from your device",
                icon: <Upload className="w-12 h-12" />
              },
              {
                step: "02",
                title: "AI Enhancement",
                description: "Our AI model processes and enhances the image in real-time",
                icon: <Brain className="w-12 h-12" />
              },
              {
                step: "03",
                title: "Download Results",
                description: "Get enhanced images with quality metrics and attention maps",
                icon: <CheckCircle className="w-12 h-12" />
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                className="relative"
              >
                <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-200 hover:shadow-xl transition-shadow">
                  <div className="text-6xl font-bold text-purple-100 mb-4">{item.step}</div>
                  <div className="text-purple-600 mb-4">{item.icon}</div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-3">{item.title}</h3>
                  <p className="text-gray-600">{item.description}</p>
                </div>
                {index < 2 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                    <ChevronRight className="w-8 h-8 text-purple-300" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Video Demo Placeholder */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="relative rounded-2xl overflow-hidden shadow-2xl max-w-5xl mx-auto"
          >
            <div className="aspect-video bg-gradient-to-br from-purple-900 via-purple-700 to-pink-700 flex items-center justify-center">
              <div className="text-center text-white">
                <div className="bg-white/20 backdrop-blur-lg rounded-full p-8 inline-block mb-4 hover:bg-white/30 transition-colors cursor-pointer">
                  <Play className="w-16 h-16" />
                </div>
                <h3 className="text-2xl font-bold mb-2">Watch Demo Video</h3>
                <p className="text-white/80">See Mantra Health in action (2:30)</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* About Us Section */}
      <section className="py-20 bg-white" id="about">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <div className="inline-flex items-center px-4 py-2 bg-pink-100 text-pink-600 rounded-full mb-4">
                <Heart className="w-4 h-4 mr-2" />
                <span className="text-sm font-semibold">About Us</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                Transforming Healthcare Through AI
              </h2>
              <p className="text-lg text-gray-600 mb-6">
                Mantra Health is dedicated to advancing medical imaging technology through artificial intelligence. Our mission is to make high-quality diagnostic imaging accessible to healthcare providers worldwide.
              </p>
              <p className="text-lg text-gray-600 mb-8">
                Built by a team of AI researchers, medical professionals, and engineers, we combine cutting-edge deep learning techniques with real-world clinical needs to deliver solutions that make a difference.
              </p>
              <div className="space-y-4">
                {[
                  "Advanced AI research and development",
                  "Collaboration with medical institutions",
                  "Commitment to data privacy and security",
                  "Continuous innovation and improvement"
                ].map((item, index) => (
                  <div key={index} className="flex items-start">
                    <CheckCircle className="w-6 h-6 text-green-500 mr-3 flex-shrink-0 mt-1" />
                    <span className="text-gray-700">{item}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="relative"
            >
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl p-8 text-white">
                  <Users className="w-12 h-12 mb-4" />
                  <div className="text-4xl font-bold mb-2">500+</div>
                  <div className="text-white/80">Healthcare Providers</div>
                </div>
                <div className="bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl p-8 text-white mt-8">
                  <Award className="w-12 h-12 mb-4" />
                  <div className="text-4xl font-bold mb-2">15+</div>
                  <div className="text-white/80">Research Papers</div>
                </div>
                <div className="bg-gradient-to-br from-green-500 to-emerald-500 rounded-2xl p-8 text-white -mt-8">
                  <Brain className="w-12 h-12 mb-4" />
                  <div className="text-4xl font-bold mb-2">99.8%</div>
                  <div className="text-white/80">AI Accuracy</div>
                </div>
                <div className="bg-gradient-to-br from-orange-500 to-red-500 rounded-2xl p-8 text-white">
                  <Shield className="w-12 h-12 mb-4" />
                  <div className="text-4xl font-bold mb-2">100%</div>
                  <div className="text-white/80">HIPAA Compliant</div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* AI Chatbot Section */}
      <section className="py-20 bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50" id="chatbot">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center px-4 py-2 bg-indigo-100 text-indigo-600 rounded-full mb-4">
              <Bot className="w-4 h-4 mr-2" />
              <span className="text-sm font-semibold">AI Assistant</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Your AI Healthcare Assistant
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Get instant answers about X-ray analysis, image quality, and medical imaging best practices
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-12 items-center mb-12">
            {/* Chatbot Features */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="space-y-6"
            >
              <div className="flex items-start space-x-4">
                <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-3 rounded-xl">
                  <MessageSquare className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    Instant Medical Insights
                  </h3>
                  <p className="text-gray-600">
                    Ask questions about X-ray quality, enhancement results, and interpretation guidance in real-time
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="bg-gradient-to-br from-blue-500 to-cyan-500 p-3 rounded-xl">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    AI-Powered Recommendations
                  </h3>
                  <p className="text-gray-600">
                    Get intelligent suggestions for optimal image enhancement settings and processing techniques
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="bg-gradient-to-br from-green-500 to-emerald-500 p-3 rounded-xl">
                  <Lightbulb className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    Educational Support
                  </h3>
                  <p className="text-gray-600">
                    Learn about medical imaging, AI enhancement technology, and best practices for diagnostic imaging
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="bg-gradient-to-br from-orange-500 to-red-500 p-3 rounded-xl">
                  <Clock className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    24/7 Availability
                  </h3>
                  <p className="text-gray-600">
                    Access expert assistance anytime, anywhere with our always-available AI chatbot
                  </p>
                </div>
              </div>

              <button
                onClick={() => setShowChat(true)}
                className="group px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-semibold text-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 inline-flex items-center"
              >
                <MessageSquare className="mr-2 w-5 h-5" />
                Try AI Assistant Now
                <ChevronRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </motion.div>

            {/* Chatbot Preview */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="relative"
            >
              <div className="bg-white rounded-2xl shadow-2xl p-6 border border-gray-200">
                <div className="flex items-center space-x-3 mb-6 pb-4 border-b">
                  <div className="relative">
                    <div className="bg-gradient-to-br from-purple-600 to-pink-600 p-3 rounded-full">
                      <Bot className="w-6 h-6 text-white" />
                    </div>
                    <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 border-2 border-white rounded-full"></div>
                  </div>
                  <div>
                    <h4 className="font-bold text-gray-900">Mantra AI Assistant</h4>
                    <p className="text-sm text-green-600">Online</p>
                  </div>
                </div>

                {/* Sample Chat Messages */}
                <div className="space-y-4 mb-4">
                  <div className="flex items-start space-x-2">
                    <div className="bg-gradient-to-br from-purple-100 to-pink-100 p-3 rounded-2xl rounded-tl-none max-w-xs">
                      <p className="text-sm text-gray-800">
                        Hi! I'm your AI healthcare assistant. How can I help you with X-ray enhancement today?
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-2 justify-end">
                    <div className="bg-gradient-to-br from-blue-500 to-cyan-500 p-3 rounded-2xl rounded-tr-none max-w-xs">
                      <p className="text-sm text-white">
                        What metrics should I look for in enhanced X-rays?
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-2">
                    <div className="bg-gradient-to-br from-purple-100 to-pink-100 p-3 rounded-2xl rounded-tl-none max-w-xs">
                      <p className="text-sm text-gray-800">
                        Focus on PSNR (Peak Signal-to-Noise Ratio) for image quality, SSIM for structural similarity, and LPIPS for perceptual quality. Higher PSNR and SSIM values indicate better enhancement!
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-2 pt-4 border-t">
                  <input
                    type="text"
                    placeholder="Ask me anything..."
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
                    disabled
                  />
                  <button className="bg-gradient-to-r from-purple-600 to-pink-600 p-2 rounded-full text-white">
                    <ChevronRight className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Floating badges */}
              <div className="absolute -top-4 -right-4 bg-white rounded-full px-4 py-2 shadow-lg border border-gray-200">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm font-semibold text-gray-700">AI Powered</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 relative overflow-hidden">
        <div className="absolute inset-0 bg-black/10"></div>
        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to Transform Your Medical Imaging?
            </h2>
            <p className="text-xl text-white/90 mb-8">
              Join hundreds of healthcare providers using Mantra Health AI
            </p>
            <button
              onClick={onGetStarted}
              className="px-8 py-4 bg-white text-purple-600 rounded-full font-semibold text-lg hover:bg-gray-100 transition-all duration-300 hover:shadow-2xl hover:scale-105 inline-flex items-center"
            >
              Start Enhancing Now
              <ChevronRight className="ml-2 w-5 h-5" />
            </button>
          </motion.div>
        </div>
      </section>

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
          className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-600 to-pink-600 text-white p-4 rounded-full shadow-2xl z-50 hover:shadow-purple-500/50 transition-all duration-300"
        >
          <MessageSquare className="h-6 w-6" />
        </motion.button>
      )}
    </div>
  );
};

export default LandingPage;
