import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams, Link } from 'react-router-dom';
import { TrendingUp, TrendingDown, AlertCircle, BarChart3, Search, Filter, ChevronLeft, ChevronRight, X, Users } from 'lucide-react';
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
    const pages = [];
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }

    return pages.map(page => (
      <button
        key={page}
        onClick={() => handlePageChange(page)}
        className={`px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
          currentPage === page
            ? 'bg-blue-600 text-white shadow-md'
            : 'text-gray-700 hover:bg-white hover:shadow-md border border-gray-300'
        }`}
      >
        {page}
      </button>
    ));
  };

  const handleStrategyClick = (strategy) => {
    navigate(`/strategy/${strategy.name}`);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading strategies...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-lg border-b border-blue-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center space-x-3">
              <div className="bg-white p-2 rounded-lg shadow-md">
                <BarChart3 className="w-6 h-6 text-blue-600" />
              </div>
              <h1 className="text-2xl font-bold text-white">Crypto Signal Lab</h1>
            </div>
            
            {/* Navigation */}
            <nav className="flex space-x-1">
              <Link
                to="/"
                className="px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-200 bg-white text-blue-600 shadow-md"
              >
                Simulator
              </Link>
              <Link
                to="/user-management"
                className="px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-200 text-blue-100 hover:text-white hover:bg-blue-500"
              >
                User Management
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <AlertCircle className="h-5 w-5 text-red-400" />
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Search and Filters */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
            <div className="flex-1 max-w-lg">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  placeholder="Search strategies..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                />
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                <Filter className="w-4 h-4" />
                <span>Filters</span>
              </button>
            </div>
          </div>

          {/* Filter Options */}
          {showFilters && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Exchange</label>
                  <select
                    value={filterExchange}
                    onChange={(e) => setFilterExchange(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  >
                    <option value="all">All Exchanges</option>
                    {getUniqueExchanges().map(exchange => (
                      <option key={exchange} value={exchange}>{exchange}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
                  <select
                    value={filterSymbol}
                    onChange={(e) => setFilterSymbol(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  >
                    <option value="all">All Symbols</option>
                    {getUniqueSymbols().map(symbol => (
                      <option key={symbol} value={symbol}>{symbol.toUpperCase()}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Timeframe</label>
                  <select
                    value={filterTimeframe}
                    onChange={(e) => setFilterTimeframe(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  >
                    <option value="all">All Timeframes</option>
                    {getUniqueTimeframes().map(timeframe => (
                      <option key={timeframe} value={timeframe}>{timeframe}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Strategies Table */}
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="px-6 py-4 bg-gradient-to-r from-gray-50 to-blue-50 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">
                Strategies ({filteredStrategies.length})
              </h2>
              <div className="text-sm text-gray-600">
                Showing {indexOfFirstRecord + 1}-{Math.min(indexOfLastRecord, filteredStrategies.length)} of {filteredStrategies.length}
              </div>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Exchange</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timeframe</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P&L</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {currentRecords.map((strategy) => (
                  <tr 
                    key={strategy.id} 
                    className="hover:bg-gray-50 transition-colors cursor-pointer"
                    onClick={() => handleStrategyClick(strategy)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{strategy.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{strategy.parameters.exchange}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{strategy.parameters.symbol.toUpperCase()}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{strategy.parameters.timeframe}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${
                        strategy.performance.profitLoss >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatCurrency(strategy.performance.profitLoss)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        {strategy.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {filteredStrategies.length === 0 && (
            <div className="text-center py-12">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No strategies found</h3>
              <p className="text-gray-500">Try adjusting your search or filter criteria.</p>
            </div>
          )}
        </div>

        {/* Pagination */}
        {filteredStrategies.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-700">
                Showing {indexOfFirstRecord + 1} to {Math.min(indexOfLastRecord, filteredStrategies.length)} of {filteredStrategies.length} results
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
                  {renderPageNumbers()}
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
      </main>
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

// User Management Component with Router
function UserManagementWithRouter() {
  return <UserManagement />;
}

// Main App Component
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<StrategyList />} />
        <Route path="/strategy/:strategyName" element={<StrategyDetailWithRouter />} />
        <Route path="/user-management" element={<UserManagementWithRouter />} />
      </Routes>
    </Router>
  );
}

export default App;
