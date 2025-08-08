import React from 'react';
import { Link } from 'react-router-dom';
import { BarChart3 } from 'lucide-react';

const Header = ({ activePage = 'simulator' }) => {
  return (
    <header className="bg-gradient-to-r from-slate-900 via-blue-900 to-slate-900 shadow-lg border-b border-blue-800/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between py-6">
          <div className="flex items-center space-x-6">
            <Link to="/" className="flex items-center space-x-6 hover:opacity-80 transition-opacity">
              <div className="relative">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg flex items-center justify-center">
                  <BarChart3 className="w-7 h-7 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white animate-pulse"></div>
              </div>
              <div className="flex flex-col">
                <h1 className="text-3xl font-bold text-white tracking-tight">Crypto Signal Lab</h1>
                <p className="text-blue-200 text-sm font-medium">Advanced Trading Analytics Platform</p>
              </div>
            </Link>
          </div>
          
          {/* Enhanced Navigation */}
          <nav className="flex items-center space-x-3">
            <Link
              to="/"
              className={`px-6 py-3 text-sm font-semibold rounded-xl transition-all duration-300 ${
                activePage === 'simulator'
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg hover:shadow-xl hover:scale-105 transform border border-blue-500/20'
                  : 'text-blue-200 hover:text-white hover:bg-white/10 backdrop-blur-sm border border-blue-500/20'
              }`}
            >
              <span>Simulator</span>
            </Link>
            <Link
              to="/user-management"
              className={`px-6 py-3 text-sm font-semibold rounded-xl transition-all duration-300 ${
                activePage === 'user-management'
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg hover:shadow-xl hover:scale-105 transform border border-blue-500/20'
                  : 'text-blue-200 hover:text-white hover:bg-white/10 backdrop-blur-sm border border-blue-500/20'
              }`}
            >
              <span>User Management</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header; 