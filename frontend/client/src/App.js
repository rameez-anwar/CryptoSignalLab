import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams, Link } from 'react-router-dom';
import { TrendingUp, TrendingDown, AlertCircle, BarChart3, Search, Filter, ChevronLeft, ChevronRight, X, Users, ChevronUp, ChevronDown } from 'lucide-react';
import StrategyDetail from './StrategyDetail';
import UserManagement from './UserManagement';
import Header from './components/Header';
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
  const [sortConfig, setSortConfig] = useState({ key: 'pnl', direction: 'desc' });
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

  const formatPercentage = (amount) => {
    return `${amount >= 0 ? '+' : ''}${amount.toFixed(2)}%`;
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

  // Sorting function
  const sortStrategies = useCallback((strategies, sortConfig) => {
    if (!sortConfig.key) return strategies;

    return [...strategies].sort((a, b) => {
      let aValue, bValue;

      switch (sortConfig.key) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'exchange':
          aValue = a.parameters.exchange.toLowerCase();
          bValue = b.parameters.exchange.toLowerCase();
          break;
        case 'symbol':
          aValue = a.parameters.symbol.toLowerCase();
          bValue = b.parameters.symbol.toLowerCase();
          break;
        case 'timeframe':
          aValue = a.parameters.timeframe.toLowerCase();
          bValue = b.parameters.timeframe.toLowerCase();
          break;
        case 'pnl':
          aValue = a.performance.profitLoss;
          bValue = b.performance.profitLoss;
          break;
        case 'status':
          aValue = a.status.toLowerCase();
          bValue = b.status.toLowerCase();
          break;
        default:
          return 0;
      }

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, []);

  // Apply sorting to filtered strategies
  const sortedStrategies = useMemo(() => {
    return sortStrategies(filteredStrategies, sortConfig);
  }, [filteredStrategies, sortConfig, sortStrategies]);

  // Handle sort
  const handleSort = (key) => {
    setSortConfig(prevConfig => ({
      key,
      direction: prevConfig.key === key && prevConfig.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  // Get sort icon
  const getSortIcon = (key) => {
    if (sortConfig.key !== key) {
      return <ChevronUp className="w-4 h-4 text-gray-400" />;
    }
    return sortConfig.direction === 'asc' 
      ? <ChevronUp className="w-4 h-4 text-blue-600" />
      : <ChevronDown className="w-4 h-4 text-blue-600" />;
  };

  // Calculate summary statistics
  const getSummaryStats = () => {
    if (strategies.length === 0) return { totalPnL: 0, avgPnL: 0, profitableCount: 0 };
    
    const totalPnL = strategies.reduce((sum, s) => sum + s.performance.profitLoss, 0);
    const avgPnL = totalPnL / strategies.length;
    const profitableCount = strategies.filter(s => s.performance.profitLoss > 0).length;
    
    return { totalPnL, avgPnL, profitableCount };
  };

  // Pagination logic
  const indexOfLastRecord = currentPage * recordsPerPage;
  const indexOfFirstRecord = indexOfLastRecord - recordsPerPage;
  const currentRecords = sortedStrategies.slice(indexOfFirstRecord, indexOfLastRecord);
  const totalPages = Math.ceil(sortedStrategies.length / recordsPerPage);

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
            : 'text-gray-700 hover:bg-gray-50 hover:shadow-sm border border-gray-200'
        }`}
      >
        {page}
      </button>
    ));
  };

  const handleStrategyClick = (strategy) => {
    navigate(`/strategy/${strategy.name}`);
  };

  const summaryStats = getSummaryStats();

    if (loading) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-gray-300 border-t-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 font-medium">Loading strategies...</p>
          <p className="text-sm text-gray-500 mt-2">Preparing your dashboard</p>
          </div>
        </div>
      );
    }

    return (
    <div className="min-h-screen bg-gray-50">
      <Header activePage="simulator" />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Professional Introduction Section */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Advanced Trading Strategy Analytics</h2>
            <p className="text-lg text-gray-600 mb-6 max-w-3xl mx-auto">
              Monitor and analyze the performance of your cryptocurrency trading strategies in real-time. 
              Track profit/loss percentages, identify profitable patterns, and optimize your trading decisions 
              with comprehensive analytics and detailed performance metrics.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <BarChart3 className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-sm font-semibold text-gray-900 mb-1">Real-time Monitoring</h3>
                <p className="text-xs text-gray-600">Live performance tracking across all strategies</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <TrendingUp className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-sm font-semibold text-gray-900 mb-1">Performance Analytics</h3>
                <p className="text-xs text-gray-600">Detailed P&L analysis and trend identification</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <Search className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-sm font-semibold text-gray-900 mb-1">Strategy Optimization</h3>
                <p className="text-xs text-gray-600">Identify and optimize profitable trading patterns</p>
              </div>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
              <div>
                <h3 className="text-sm font-medium text-red-800">Connection Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Search and Filters */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
            <div className="flex-1 max-w-lg">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  placeholder="Search strategies by name, exchange, or symbol..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="block w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                />
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  showFilters || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Filter className="w-4 h-4" />
                <span>Filters</span>
                {(filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all') && (
                  <span className="bg-red-500 text-white text-xs rounded-full px-2 py-1">
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
          
          {/* Filter Options */}
          {showFilters && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Exchange</label>
                  <select
                    value={filterExchange}
                    onChange={(e) => setFilterExchange(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
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
        </div>

        {/* Enhanced Strategies Table */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 bg-gradient-to-r from-gray-50 to-blue-50 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-gray-900">Trading Strategies</h2>
                <p className="text-sm text-gray-600 mt-1">Performance monitoring and analysis</p>
              </div>
              <div className="flex items-center space-x-4">
                <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                  {sortedStrategies.length} Active
                </div>
                <div className="text-sm text-gray-600">
                  Showing {indexOfFirstRecord + 1}-{Math.min(indexOfLastRecord, sortedStrategies.length)} of {sortedStrategies.length}
                </div>
              </div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead className="bg-gradient-to-r from-gray-100 to-blue-100">
                <tr>
                  <th 
                    className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-blue-200 transition-colors"
                    onClick={() => handleSort('name')}
                  >
                    <div className="flex items-center space-x-2">
                      <span>Strategy</span>
                      {getSortIcon('name')}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-blue-200 transition-colors"
                    onClick={() => handleSort('exchange')}
                  >
                    <div className="flex items-center space-x-2">
                      <span>Exchange</span>
                      {getSortIcon('exchange')}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-blue-200 transition-colors"
                    onClick={() => handleSort('symbol')}
                  >
                    <div className="flex items-center space-x-2">
                      <span>Symbol</span>
                      {getSortIcon('symbol')}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-blue-200 transition-colors"
                    onClick={() => handleSort('timeframe')}
                  >
                    <div className="flex items-center space-x-2">
                      <span>Timeframe</span>
                      {getSortIcon('timeframe')}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-blue-200 transition-colors"
                    onClick={() => handleSort('pnl')}
                  >
                    <div className="flex items-center space-x-2">
                      <span>P&L</span>
                      {getSortIcon('pnl')}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-blue-200 transition-colors"
                    onClick={() => handleSort('status')}
                  >
                    <div className="flex items-center space-x-2">
                      <span>Status</span>
                      {getSortIcon('status')}
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {currentRecords.map((strategy) => (
                  <tr 
                    key={strategy.id} 
                    className="hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 transition-all duration-300 transform hover:scale-[1.01] hover:shadow-md cursor-pointer"
                    onClick={() => handleStrategyClick(strategy)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div>
                      <div className="text-sm font-semibold text-gray-900">{strategy.name}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 bg-gray-50 px-3 py-1 rounded-full inline-block">
                        {strategy.parameters.exchange.toUpperCase()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 bg-blue-50 px-3 py-1 rounded-full inline-block">
                        {strategy.parameters.symbol.toUpperCase()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 bg-green-50 px-3 py-1 rounded-full inline-block">
                        {strategy.parameters.timeframe}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        {strategy.performance.profitLoss >= 0 ? (
                          <div className="flex items-center space-x-2">
                            <TrendingUp className="w-4 h-4 text-green-600" />
                            <span className="text-sm font-bold text-green-600 bg-green-50 px-3 py-1 rounded-full">
                              {formatPercentage(strategy.performance.profitLoss)}
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <TrendingDown className="w-4 h-4 text-red-600" />
                            <span className="text-sm font-bold text-red-600 bg-red-50 px-3 py-1 rounded-full">
                              {formatPercentage(strategy.performance.profitLoss)}
                            </span>
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold ${
                        strategy.status === 'Active' 
                          ? 'bg-green-100 text-green-800 border border-green-200' 
                          : 'bg-red-100 text-red-800 border border-red-200'
                      }`}>
                        <div className={`w-2 h-2 rounded-full mr-2 ${
                          strategy.status === 'Active' ? 'bg-green-500' : 'bg-red-500'
                        }`}></div>
                        {strategy.status}
                            </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {sortedStrategies.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <BarChart3 className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                {searchTerm || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all' 
                  ? 'No strategies match your filters' 
                  : 'No strategies found'}
              </h3>
              <p className="text-gray-500 max-w-md mx-auto">
                {searchTerm || filterExchange !== 'all' || filterSymbol !== 'all' || filterTimeframe !== 'all'
                  ? 'Try adjusting your search or filter criteria to find matching strategies.'
                  : 'No strategies are currently configured in the database.'}
              </p>
            </div>
          )}
        </div>

        {/* Enhanced Pagination */}
          {sortedStrategies.length > 0 && totalPages > 1 && (
          <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6 mt-6">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-700">
                Showing <span className="font-medium text-blue-600">{indexOfFirstRecord + 1}</span> to <span className="font-medium text-blue-600">{Math.min(indexOfLastRecord, sortedStrategies.length)}</span> of <span className="font-medium text-blue-600">{sortedStrategies.length}</span> results
                </div>
              
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handlePreviousPage}
                    disabled={currentPage === 1}
                  className="px-4 py-2 text-sm font-medium rounded-lg text-gray-700 hover:bg-gray-50 hover:shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2 border border-gray-200"
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
                  className="px-4 py-2 text-sm font-medium rounded-lg text-gray-700 hover:bg-gray-50 hover:shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2 border border-gray-200"
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
