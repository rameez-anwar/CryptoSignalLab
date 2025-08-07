import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from 'react-router-dom';
import { TrendingUp, TrendingDown, Activity, AlertCircle, BarChart3, Search, Filter, Users, ChevronLeft, ChevronRight, X } from 'lucide-react';
import StrategyDetail from './StrategyDetail';
import UserManagement from './UserManagement';
import './App.css';

// Strategy List Component
function StrategyList() {
  const [strategies, setStrategies] = useState([]);
  const [filteredStrategies, setFilteredStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterExchange, setFilterExchange] = useState('all');
  const [filterSymbol, setFilterSymbol] = useState('all');
  const [filterTimeframe, setFilterTimeframe] = useState('all');
  const [activeTab, setActiveTab] = useState('simulator');
  const [currentPage, setCurrentPage] = useState(1);
  const [recordsPerPage] = useState(10);
  const [showFilters, setShowFilters] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchStrategies();
  }, []);

  const filterStrategies = useCallback(() => {
    let filtered = strategies;
    if (searchTerm) {
      filtered = filtered.filter(strategy =>
        strategy.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        strategy.parameters.exchange.toLowerCase().includes(searchTerm.toLowerCase()) ||
        strategy.parameters.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    if (filterExchange !== 'all') {
      filtered = filtered.filter(strategy => strategy.parameters.exchange === filterExchange);
    }
    if (filterSymbol !== 'all') {
      filtered = filtered.filter(strategy => strategy.parameters.symbol === filterSymbol);
    }
    if (filterTimeframe !== 'all') {
      filtered = filtered.filter(strategy => strategy.parameters.timeframe === filterTimeframe);
    }
    setFilteredStrategies(filtered);
    setCurrentPage(1); // Reset to first page when filters change
  }, [strategies, searchTerm, filterExchange, filterSymbol, filterTimeframe]);

  useEffect(() => {
    filterStrategies();
  }, [strategies, searchTerm, filterExchange, filterSymbol, filterTimeframe, filterStrategies]);

  const fetchStrategies = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/strategies');
      setStrategies(response.data.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch strategies. Please make sure the server is running and database is connected.');
      console.error('Error fetching strategies:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(amount);
  };

  const getStatusColor = (status) => {
    return status === 'Active' ? 'text-green-600' : 'text-red-600';
  };

  const getStatusIcon = (status) => {
    return status === 'Active' ? <Activity className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />;
  };

  const getPerformanceColor = (value) => {
    return value >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const getUniqueExchanges = () => {
    return [...new Set(strategies.map(s => s.parameters.exchange))];
  };

  const getUniqueSymbols = () => {
    return [...new Set(strategies.map(s => s.parameters.symbol))];
  };

  const getUniqueTimeframes = () => {
    return [...new Set(strategies.map(s => s.parameters.timeframe))];
  };

  // Pagination logic
  const indexOfLastRecord = currentPage * recordsPerPage;
  const indexOfFirstRecord = indexOfLastRecord - recordsPerPage;
  const currentRecords = filteredStrategies.slice(indexOfFirstRecord, indexOfLastRecord);
  const totalPages = Math.ceil(filteredStrategies.length / recordsPerPage);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  const renderPageNumbers = () => {
    const pageNumbers = [];
    const maxVisiblePages = 5;
    
    if (totalPages <= maxVisiblePages) {
      // Show all pages if total pages is less than or equal to max visible
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i);
      }
    } else {
      // Show limited pages with ellipsis
      if (currentPage <= 3) {
        // Show first 3 pages + ellipsis + last page
        for (let i = 1; i <= 3; i++) {
          pageNumbers.push(i);
        }
        pageNumbers.push('...');
        pageNumbers.push(totalPages);
      } else if (currentPage >= totalPages - 2) {
        // Show first page + ellipsis + last 3 pages
        pageNumbers.push(1);
        pageNumbers.push('...');
        for (let i = totalPages - 2; i <= totalPages; i++) {
          pageNumbers.push(i);
        }
      } else {
        // Show first page + ellipsis + current page + ellipsis + last page
        pageNumbers.push(1);
        pageNumbers.push('...');
        pageNumbers.push(currentPage);
        pageNumbers.push('...');
        pageNumbers.push(totalPages);
      }
    }
    
    return pageNumbers;
  };

  const handleStrategyClick = (strategy) => {
    navigate(`/strategy/${strategy.name}`);
  };

  const renderSimulatorContent = () => {
    if (loading) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading simulator strategies...</p>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="text-center max-w-md mx-auto p-6">
            <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Connection Error</h2>
            <p className="text-gray-600 mb-4">{error}</p>
            <button
              onClick={fetchStrategies}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      );
    }

    return (
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search and Filter Section */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl shadow-lg border border-blue-100 p-6 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
            <div className="flex-1 max-w-md">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-blue-500" />
                </div>
                <input
                  type="text"
                  placeholder="Search strategies by name, exchange, or symbol..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="block w-full pl-10 pr-3 py-3 border border-blue-200 rounded-lg leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm transition-all duration-200"
                />
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  showFilters || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all'
                    ? 'bg-blue-600 text-white shadow-md hover:bg-blue-700'
                    : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50 hover:border-blue-300'
                }`}
              >
                <Filter className="h-4 w-4" />
                <span>Filters</span>
                {(filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all') && (
                  <span className="bg-red-500 text-white text-xs rounded-full px-2 py-0.5">
                    {[filterExchange, filterSymbol, filterTimeframe].filter(f => f !== 'all').length}
                  </span>
                )}
              </button>
              
              {(searchTerm || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all') && (
                <button
                  onClick={() => {
                    setSearchTerm('');
                    setFilterExchange('all');
                    setFilterSymbol('all');
                    setFilterTimeframe('all');
                    setShowFilters(false);
                  }}
                  className="text-sm text-blue-600 hover:text-blue-800 font-medium px-3 py-2 rounded-lg hover:bg-blue-50 transition-colors"
                >
                  Clear all
                </button>
              )}
            </div>
          </div>
          
          {/* Filter Modal */}
          {showFilters && (
            <div className="mt-4 p-4 bg-white rounded-lg border border-gray-200 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Advanced Filters</h3>
                <button
                  onClick={() => setShowFilters(false)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Exchange</label>
                  <select
                    value={filterExchange}
                    onChange={(e) => setFilterExchange(e.target.value)}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                  >
                    <option value="all">All Exchanges</option>
                    {getUniqueExchanges().map(exchange => (
                      <option key={exchange} value={exchange}>{exchange.toUpperCase()}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Symbol</label>
                  <select
                    value={filterSymbol}
                    onChange={(e) => setFilterSymbol(e.target.value)}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                  >
                    <option value="all">All Symbols</option>
                    {getUniqueSymbols().map(symbol => (
                      <option key={symbol} value={symbol}>{symbol.toUpperCase()}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Timeframe</label>
                  <select
                    value={filterTimeframe}
                    onChange={(e) => setFilterTimeframe(e.target.value)}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                  >
                    <option value="all">All Timeframes</option>
                    {getUniqueTimeframes().map(timeframe => (
                      <option key={timeframe} value={timeframe}>{timeframe.toUpperCase()}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}
          
          {(searchTerm || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all') && (
            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <span>Results:</span>
                  <span className="font-semibold text-blue-600">{filteredStrategies.length}</span>
                  <span>strategies</span>
                  <span className="text-gray-400">|</span>
                  <span>Page</span>
                  <span className="font-semibold text-blue-600">{currentPage}</span>
                  <span>of</span>
                  <span className="font-semibold text-blue-600">{totalPages}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Strategies Table */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 bg-gradient-to-r from-gray-50 to-blue-50 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">Trading Strategies</h2>
                <p className="text-gray-600 mt-1">Real-time performance monitoring and analysis</p>
              </div>
              <div className="flex items-center space-x-2">
                <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                  {filteredStrategies.length} Active
                </div>
              </div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gradient-to-r from-blue-600 to-indigo-600">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-white uppercase tracking-wider">
                    Strategy Name
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-white uppercase tracking-wider">
                    Exchange
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-white uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-white uppercase tracking-wider">
                    Timeframe
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-white uppercase tracking-wider">
                    P&L
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-white uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {currentRecords.map((strategy) => (
                  <tr 
                    key={strategy.id} 
                    className="hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 transition-all duration-200 cursor-pointer"
                    onClick={() => handleStrategyClick(strategy)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-semibold text-gray-900">{strategy.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {strategy.parameters.exchange.toUpperCase()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        {strategy.parameters.symbol.toUpperCase()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{strategy.parameters.timeframe}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        {strategy.performance.profitLoss >= 0 ? (
                          <div className="flex items-center space-x-1">
                            <TrendingUp className="w-4 h-4 text-green-600" />
                            <span className={`text-sm font-semibold text-green-600`}>
                              {formatCurrency(strategy.performance.profitLoss)}
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-1">
                            <TrendingDown className="w-4 h-4 text-red-600" />
                            <span className={`text-sm font-semibold text-red-600`}>
                              {formatCurrency(strategy.performance.profitLoss)}
                            </span>
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        {strategy.status === 'Active' ? (
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                              Active
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                              Inactive
                            </span>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredStrategies.length === 0 && (
            <div className="text-center py-12">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                {searchTerm || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all' 
                  ? 'No strategies match your filters' 
                  : 'No strategies found'}
              </h3>
              <p className="text-gray-500">
                {searchTerm || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all'
                  ? 'Try adjusting your search or filter criteria.'
                  : 'No strategies are currently configured in the database.'}
              </p>
            </div>
          )}

          {/* Pagination Controls */}
          {filteredStrategies.length > 0 && totalPages > 1 && (
            <div className="px-6 py-4 bg-gradient-to-r from-gray-50 to-blue-50 border-t border-gray-200">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-700">
                  Showing <span className="font-semibold text-blue-600">{indexOfFirstRecord + 1}</span> to <span className="font-semibold text-blue-600">{Math.min(indexOfLastRecord, filteredStrategies.length)}</span> of <span className="font-semibold text-blue-600">{filteredStrategies.length}</span> results
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handlePreviousPage}
                    disabled={currentPage === 1}
                    className="px-4 py-2 text-sm font-medium rounded-lg text-gray-700 hover:bg-white hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center space-x-2 border border-gray-300"
                  >
                    <ChevronLeft className="h-4 w-4" />
                    <span>Previous</span>
                  </button>
                  
                  <div className="flex items-center space-x-1">
                    {renderPageNumbers().map((page, index) => (
                      <button
                        key={index}
                        onClick={() => typeof page === 'number' && handlePageChange(page)}
                        disabled={page === '...'}
                        className={`px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                          page === currentPage
                            ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md'
                            : page === '...'
                            ? 'text-gray-400 cursor-default'
                            : 'text-gray-700 hover:bg-white hover:shadow-md border border-gray-300'
                        }`}
                      >
                        {page}
                      </button>
                    ))}
                  </div>
                  
                  <button
                    onClick={handleNextPage}
                    disabled={currentPage === totalPages}
                    className="px-4 py-2 text-sm font-medium rounded-lg text-gray-700 hover:bg-white hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center space-x-2 border border-gray-300"
                  >
                    <span>Next</span>
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    );
  };

  const renderUserManagementContent = () => {
    return <UserManagement />;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-lg border-b border-blue-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="bg-white p-2 rounded-lg shadow-md">
                <BarChart3 className="w-6 h-6 text-blue-600" />
              </div>
              <h1 className="text-2xl font-bold text-white">Crypto Signal Lab</h1>
            </div>
            
            {/* Navigation Tabs */}
            <nav className="flex space-x-1 ml-8">
              <button
                onClick={() => setActiveTab('simulator')}
                className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${
                  activeTab === 'simulator'
                    ? 'bg-white text-blue-600 shadow-md'
                    : 'text-blue-100 hover:text-white hover:bg-blue-500'
                }`}
              >
                Simulator
              </button>
              <button
                onClick={() => setActiveTab('user-management')}
                className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${
                  activeTab === 'user-management'
                    ? 'bg-white text-blue-600 shadow-md'
                    : 'text-blue-100 hover:text-white hover:bg-blue-500'
                }`}
              >
                User Management
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Content based on active tab */}
      {activeTab === 'simulator' ? renderSimulatorContent() : renderUserManagementContent()}
    </div>
  );
}

// Strategy Detail Component with Router
function StrategyDetailWithRouter() {
  const { strategyName } = useParams();
  const navigate = useNavigate();

  const handleBack = () => {
    navigate('/');
  };

  return <StrategyDetail strategyName={strategyName} onBack={handleBack} />;
}

// Main App Component
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<StrategyList />} />
        <Route path="/strategy/:strategyName" element={<StrategyDetailWithRouter />} />
      </Routes>
    </Router>
  );
}

export default App;
