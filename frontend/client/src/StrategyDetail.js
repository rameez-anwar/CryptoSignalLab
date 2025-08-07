import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import { ArrowLeft, TrendingUp, TrendingDown, BarChart3, AlertCircle } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
} from '@tanstack/react-table';

function StrategyDetail({ strategyName, onBack }) {
  const [strategy, setStrategy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pnlSeries, setPnlSeries] = useState([]);
  const [winLossData, setWinLossData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [ledgerData, setLedgerData] = useState(null);
  const [ledgerLoading, setLedgerLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [totalItems, setTotalItems] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  // TanStack Table column helper
  const columnHelper = createColumnHelper();

  // Define columns for TanStack Table - same columns as before
  const columns = useMemo(
    () => [
      columnHelper.accessor('id', {
        header: '#',
        cell: (info) => `#${info.getValue()}`,
        size: 60,
      }),
      columnHelper.accessor('time', {
        header: 'Time',
        cell: (info) => info.getValue(),
        size: 150,
      }),
      columnHelper.accessor('action', {
        header: 'Action',
        cell: (info) => {
          const action = info.getValue();
          const getActionStyle = (action) => {
            switch (action) {
              case 'buy':
                return 'bg-gradient-to-r from-green-400 to-emerald-500 text-white shadow-lg hover:shadow-xl transform hover:scale-105';
              case 'sell':
                return 'bg-gradient-to-r from-red-400 to-pink-500 text-white shadow-lg hover:shadow-xl transform hover:scale-105';
              case 'take_profit':
                return 'bg-gradient-to-r from-blue-400 to-indigo-500 text-white shadow-lg hover:shadow-xl transform hover:scale-105';
              case 'stop_loss':
                return 'bg-gradient-to-r from-orange-400 to-yellow-500 text-white shadow-lg hover:shadow-xl transform hover:scale-105';
              default:
                return 'bg-gradient-to-r from-gray-400 to-gray-500 text-white shadow-lg hover:shadow-xl transform hover:scale-105';
            }
          };
          return (
            <span className={`inline-flex px-3 py-1 text-xs font-bold rounded-full transition-all duration-200 ${getActionStyle(action)}`}>
              {action.toUpperCase()}
            </span>
          );
        },
        size: 100,
      }),
      columnHelper.accessor('buyPrice', {
        header: 'Buy Price',
        cell: (info) => `$${parseFloat(info.getValue() || 0).toFixed(2)}`,
        size: 120,
      }),
      columnHelper.accessor('sellPrice', {
        header: 'Sell Price',
        cell: (info) => `$${parseFloat(info.getValue() || 0).toFixed(2)}`,
        size: 120,
      }),
      columnHelper.accessor('pnlPercent', {
        header: 'PNL Percent',
        cell: (info) => {
          const value = parseFloat(info.getValue());
          const isPositive = value >= 0;
          return (
            <span className={`font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {isPositive ? '+' : ''}{value.toFixed(2)}%
            </span>
          );
        },
        size: 120,
      }),
      columnHelper.accessor('pnlSum', {
        header: 'PNL Sum',
        cell: (info) => {
          const value = parseFloat(info.getValue());
          const isPositive = value >= 0;
          return (
            <span className={`font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {isPositive ? '+' : ''}{value.toFixed(2)}%
            </span>
          );
        },
        size: 120,
      }),
      columnHelper.accessor('balance', {
        header: 'Balance',
        cell: (info) => `$${parseFloat(info.getValue()).toFixed(2)}`,
        size: 120,
      }),
    ],
    []
  );

  const fetchStrategyDetails = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/strategies/${strategyName}/details`);
      setStrategy(response.data.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch strategy details. Please try again.');
      console.error('Error fetching strategy details:', err);
    } finally {
      setLoading(false);
    }
  }, [strategyName]);

  const fetchPnlData = useCallback(async () => {
    try {
      console.log('Fetching PNL data for strategy:', strategyName);
      const response = await axios.get(`/api/strategies/${strategyName}/pnl_timeseries`);
      console.log('PNL response:', response.data);
      if (response.data.success) {
        setPnlSeries(response.data.data);
      }
    } catch (err) {
      console.error('Error fetching PNL data:', err);
    }
  }, [strategyName]);

  const fetchWinLossData = useCallback(async () => {
    try {
      console.log('Fetching win/loss data for strategy:', strategyName);
      const response = await axios.get(`/api/strategies/${strategyName}/winloss`);
      console.log('Win/Loss response:', response.data);
      if (response.data.success) {
        setWinLossData(response.data.data);
      }
    } catch (err) {
      console.error('Error fetching win/loss data:', err);
    }
  }, [strategyName]);

  const fetchMetricsData = useCallback(async () => {
    try {
      console.log('Fetching comprehensive metrics for strategy:', strategyName);
      const response = await axios.get(`/api/strategies/${strategyName}/metrics`);
      console.log('Metrics response:', response.data);
      if (response.data.success) {
        setMetricsData(response.data.data);
      }
    } catch (err) {
      console.error('Error fetching metrics data:', err);
    }
  }, [strategyName]);

  const fetchLedgerData = useCallback(async () => {
    try {
      setLedgerLoading(true);
      console.log('Fetching ledger data for strategy:', strategyName);
      const response = await axios.get(`/api/strategies/${strategyName}/ledger`, {
        params: {
          page: 1,
          limit: 10000 // Fetch large amount for client-side virtualization
        }
      });
      
      console.log('Ledger response:', response.data);
      if (response.data.success) {
        setLedgerData(response.data.data.ledger || []);
        setTotalItems(response.data.data.ledger?.length || 0);
        setTotalPages(Math.ceil((response.data.data.ledger?.length || 0) / itemsPerPage));
      }
    } catch (err) {
      console.error('Error fetching ledger data:', err);
      setLedgerData([]);
      setTotalItems(0);
      setTotalPages(0);
    } finally {
      setLedgerLoading(false);
    }
  }, [strategyName, itemsPerPage]);

  // TanStack Table instance
  const table = useReactTable({
    data: ledgerData || [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    initialState: {
      pagination: {
        pageSize: itemsPerPage,
      },
    },
  });

  useEffect(() => {
    fetchStrategyDetails();
    fetchPnlData();
    fetchWinLossData();
    fetchMetricsData();
    fetchLedgerData(); // Call fetchLedgerData here
    
    // Remove focus outlines from all SVG elements
    const style = document.createElement('style');
    style.textContent = `
      svg:focus, svg *:focus {
        outline: none !important;
      }
      .recharts-wrapper:focus, .recharts-wrapper *:focus {
        outline: none !important;
      }
      .recharts-surface:focus {
        outline: none !important;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, [fetchStrategyDetails, fetchPnlData, fetchWinLossData, fetchMetricsData, fetchLedgerData]);

  const getPerformanceColor = (value) => {
    return value >= 0 ? 'text-green-600' : 'text-red-600';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading strategy details...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto p-6">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={onBack}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  if (!strategy) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Strategy Not Found</h2>
          <p className="text-gray-600 mb-4">The requested strategy could not be found.</p>
          <button
            onClick={onBack}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-lg border-b border-blue-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center py-4">
            <button
              onClick={onBack}
              className="flex items-center space-x-2 text-blue-100 hover:text-white transition-colors mr-6"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Strategies</span>
            </button>
            <div className="flex items-center space-x-3">
              <div className="bg-white p-2 rounded-lg shadow-md">
                <BarChart3 className="w-6 h-6 text-blue-600" />
              </div>
              <h1 className="text-2xl font-bold text-white">Simulator {'>'} Strategy</h1>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Top Section: Single Horizontal Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 flex flex-col lg:flex-row gap-6 items-stretch">
          {/* Left: Strategy Info */}
          <div className="flex flex-col min-w-[200px] max-w-[250px] flex-shrink-0 border-r border-gray-100 pr-6">
            <div className="mb-4">
              <div className="flex items-center gap-3 mb-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <h2 className="text-2xl font-bold text-gray-900">{strategy.name}</h2>
              </div>
              <div className="flex flex-wrap gap-1 ml-6">
                <div className="flex items-center space-x-1 bg-gray-50 rounded px-1.5 py-0.5 border border-gray-200">
                  <svg className="w-2.5 h-2.5 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-xs text-gray-600">{strategy.parameters.general.time_horizon}</span>
                </div>
                <div className="flex items-center space-x-1 bg-gray-50 rounded px-1.5 py-0.5 border border-gray-200">
                  <svg className="w-3 h-3 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
                  </svg>
                  <span className="text-xs text-gray-600">{strategy.parameters.general.symbol}</span>
                </div>
                <div className="flex items-center space-x-1 bg-gray-50 rounded px-1.5 py-0.5 border border-gray-200">
                  <svg className="w-2.5 h-2.5 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                  <span className="text-xs text-gray-600">{strategy.parameters.general.data_exchange}</span>
                </div>
              </div>
            </div>
            
            <div className="flex gap-2 mb-4 ml-6">
              {strategy.parameters.live.take_profit && (
                <div className="flex items-center space-x-1 bg-gray-50 rounded px-2 py-1 border border-gray-200 flex-1">
                  <span className="text-xs text-gray-500">TP</span>
                  <span className="text-xs font-semibold text-green-600">{(strategy.parameters.live.take_profit * 100).toFixed(2)}%</span>
                </div>
              )}
              {strategy.parameters.live.stop_loss && (
                <div className="flex items-center space-x-1 bg-gray-50 rounded px-2 py-1 border border-gray-200 flex-1">
                  <span className="text-xs text-gray-500">SL</span>
                  <span className="text-xs font-semibold text-red-600">{(strategy.parameters.live.stop_loss * 100).toFixed(2)}%</span>
                </div>
              )}
            </div>
            
            <div className="space-y-2 ml-6">
              <div className="flex justify-between items-center bg-white rounded px-3 py-2 border border-gray-200 shadow-sm">
                <span className="text-xs text-gray-600">Total Return</span>
                <span className="text-xs font-bold text-green-600">{strategy.performance.totalReturn.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between items-center bg-white rounded px-3 py-2 border border-gray-200 shadow-sm">
                <span className="text-xs text-gray-600">Total Trades</span>
                <span className="text-xs font-bold text-blue-600">{strategy.performance.totalTrades}</span>
              </div>
            </div>
          </div>

          {/* Center: Current Status */}
          <div className="flex-1 flex flex-col justify-center items-center px-4 border-r border-gray-100">
            <h3 className="text-sm font-semibold text-gray-900 mb-2 text-center">Current Status</h3>
            <div className="grid grid-cols-3 gap-3 w-full max-w-md">
              <div className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Entry Price</div>
                <div className="text-base font-semibold text-gray-900">${parseFloat(strategy.performance.entryPrice).toFixed(2)}</div>
                </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Current Price</div>
                <div className="text-base font-semibold text-gray-900">${parseFloat(strategy.performance.currentPrice).toFixed(2)}</div>
            </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Current PNL</div>
                <div className={`text-base font-semibold ${getPerformanceColor(strategy.performance.currentPnl)}`}>{parseFloat(strategy.performance.currentPnl).toFixed(2)}%</div>
          </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Forecast</div>
                <div className="text-base font-semibold text-gray-900">{strategy.forecast.forecast || '-'}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Next Forecast</div>
                <div className="text-base font-semibold text-gray-900">{strategy.forecast.nextForecast}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Forecast Time</div>
                <div className="text-base font-semibold text-gray-900">{strategy.forecast.forecastTime || '-'}</div>
              </div>
            </div>
          </div>

          {/* Right: Historical Performance */}
          <div className="flex flex-col justify-between min-w-[220px] max-w-[260px] flex-shrink-0 pl-6">
            <h3 className="text-sm font-semibold text-gray-900 mb-2">Historical Performance</h3>
            <div className="grid grid-cols-3 gap-2 mb-2">
              {Object.entries(strategy.performance.historicalReturns).map(([period, value]) => (
                <div key={period} className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                  <div className="text-xs text-gray-500 mb-1">{period}</div>
                  <div className={`text-base font-semibold ${getPerformanceColor(value)}`}>{value.toFixed(2)}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Parameters Section */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Strategy Parameters</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* General */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">General</h4>
              <div className="space-y-2 text-xs">
                {Object.entries(strategy.parameters.general).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-600">{key}</span>
                    <span className="font-medium text-gray-900">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Filters */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Filters</h4>
              <div className="space-y-2 text-xs">
                {Object.entries(strategy.parameters.filters).map(([key, value]) => (
                  key !== 'percent_required' ? (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-600">{key}</span>
                    <span className="font-medium text-gray-900">{String(value)}</span>
                  </div>
                  ) : null
                ))}
                {'percent_required' in strategy.parameters.filters && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">percent_required</span>
                    <span className="font-medium text-gray-900">{strategy.parameters.filters.percent_required}</span>
                  </div>
                )}
              </div>
            </div>
            {/* Patterns Use */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Patterns Use</h4>
              <div className="space-y-2 text-xs">
                {Object.entries(strategy.parameters.patterns).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-600">{key}</span>
                    <span className="font-medium text-gray-900">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Dynamic Indicator Cards */}
            {strategy.parameters.indicators && Object.entries(strategy.parameters.indicators).map(([category, indicators]) => (
              <div key={category} className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-3 text-sm capitalize">
                  {category.replace(/([A-Z])/g, ' $1').trim()}
                </h4>
                <div className="space-y-2 text-xs">
                  {Object.entries(indicators).map(([indicator, windowSize]) => (
                    <div key={indicator} className="flex justify-between">
                      <span className="text-gray-600">{indicator}_window</span>
                      <span className="font-medium text-gray-900">{windowSize}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            {/* Live */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Live</h4>
              <div className="space-y-2 text-xs">
                {Object.entries(strategy.parameters.live).map(([key, value]) => {
                  // Format TP/SL values as percentages
                  let displayValue = String(value);
                  if (key === 'take_profit' && value !== null && value !== undefined) {
                    displayValue = `${(value * 100).toFixed(2)}%`;
                  } else if (key === 'stop_loss' && value !== null && value !== undefined) {
                    displayValue = `${(value * 100).toFixed(2)}%`;
                  }
                  
                  return (
                  <div key={key} className="flex justify-between">
                      <span className="text-gray-600">{key.replace(/_/g, ' ')}</span>
                      <span className="font-medium text-gray-900">{displayValue}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-white via-gray-50 to-blue-50 rounded-2xl shadow-2xl p-8 mt-8 border border-gray-100">
          <div className="flex items-center space-x-4 mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-1">Cumulative P&L Performance</h2>
              <p className="text-gray-600">Real-time profit and loss tracking over time</p>
            </div>
          </div>
          
          {pnlSeries.length > 0 ? (
            <div className="relative focus:outline-none" tabIndex="-1" style={{ outline: 'none' }}>
              <ResponsiveContainer width="100%" height={400} className="focus:outline-none" style={{ outline: 'none' }}>
                <AreaChart 
                  data={pnlSeries} 
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                  className="focus:outline-none"
                  style={{ outline: 'none' }}
                >
                  <defs>
                    <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.8}/>
                      <stop offset="50%" stopColor="#34d399" stopOpacity={0.6}/>
                      <stop offset="100%" stopColor="#6ee7b7" stopOpacity={0.3}/>
                    </linearGradient>
                    <linearGradient id="strokePnl" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#059669"/>
                      <stop offset="100%" stopColor="#10b981"/>
                    </linearGradient>
                  </defs>
                  
                  <CartesianGrid 
                    strokeDasharray="3 3" 
                    stroke="#e5e7eb" 
                    opacity={0.3}
                  />
                  
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={str => str && new Date(str).toLocaleDateString('en-US', { 
                      month: 'short', 
                      day: 'numeric' 
                    })}
                    tick={{ fontSize: 12, fill: '#6b7280' }}
                    axisLine={{ stroke: '#d1d5db', strokeWidth: 1 }}
                    tickLine={false}
                  />
                  
                  <YAxis 
                    tick={{ fontSize: 12, fill: '#6b7280' }}
                    axisLine={{ stroke: '#d1d5db', strokeWidth: 1 }}
                    tickLine={false}
                    tickFormatter={(value) => `${parseFloat(value).toFixed(2)}%`}
                  />
                  
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
                      fontSize: '12px'
                    }}
                    labelStyle={{ fontWeight: 'bold', color: '#374151' }}
                    formatter={(value, name) => [`${parseFloat(value).toFixed(2)}%`, 'P&L']}
                    labelFormatter={(label) => new Date(label).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  />
                  
                  <Area 
                    type="monotone" 
                    dataKey="pnl" 
                    stroke="url(#strokePnl)" 
                    strokeWidth={3}
                    fill="url(#colorPnl)" 
                    fillOpacity={1}
                    dot={false}
                    activeDot={{
                      r: 6,
                      fill: '#10b981',
                      stroke: '#ffffff',
                      strokeWidth: 2,
                      style: { filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }
                    }}
                  />
                </AreaChart>
              </ResponsiveContainer>
              
              {/* Performance Stats */}
              <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
                  <div className="text-sm text-gray-500 mb-1">Peak P&L</div>
                  <div className="text-xl font-bold text-green-600">
                    {Math.max(...pnlSeries.map(d => d.pnl || 0)).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
                  <div className="text-sm text-gray-500 mb-1">Data Points</div>
                  <div className="text-xl font-bold text-blue-600">
                    {pnlSeries.length}
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
                  <div className="text-sm text-gray-500 mb-1">Avg P&L</div>
                  <div className="text-xl font-bold text-purple-600">
                    {(pnlSeries.reduce((sum, d) => sum + (d.pnl || 0), 0) / pnlSeries.length).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-80">
              <div className="text-center">
                <div className="w-20 h-20 bg-gradient-to-br from-gray-100 to-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-10 h-10 text-gray-400" />
                </div>
                <h4 className="text-lg font-semibold text-gray-700 mb-2">No Performance Data</h4>
                <p className="text-gray-500 text-sm max-w-md">
                  P&L data will appear here once backtest results are available for this strategy
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Win/Loss Section */}
        <div className="bg-gradient-to-br from-white via-gray-50 to-red-50 rounded-2xl shadow-2xl p-8 mt-8 border border-gray-100">
          <div className="flex items-center space-x-4 mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-1">Win/Loss Analysis</h2>
              <p className="text-gray-600">Take Profit vs Stop Loss performance breakdown</p>
            </div>
          </div>
          
          {winLossData && winLossData.total > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-96">
              {/* Individual PNL Bar Chart */}
              <div className="flex flex-col h-full focus:outline-none" tabIndex="-1" style={{ outline: 'none' }}>
                <h4 className="text-lg font-semibold text-gray-900 mb-3 text-center">Individual P&L Distribution</h4>
                <div className="flex-1 min-h-0 focus:outline-none" tabIndex="-1" style={{ outline: 'none' }}>
                  <ResponsiveContainer width="100%" height="100%" className="focus:outline-none" style={{ outline: 'none' }}>
                    <BarChart 
                      data={winLossData?.individualPnl?.map((pnl, index) => ({ 
                        index, 
                        pnl: pnl,
                        tradeNumber: index + 1
                      })) || []}
                      margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
                      className="focus:outline-none"
                      style={{ outline: 'none' }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.3} />
                      <XAxis 
                        dataKey="index" 
                        tick={{ fontSize: 10, fill: '#6b7280' }}
                        tickLine={false}
                        axisLine={{ stroke: '#d1d5db', strokeWidth: 1 }}
                      />
                      <YAxis 
                        tick={{ fontSize: 10, fill: '#6b7280' }}
                        tickLine={false}
                        axisLine={{ stroke: '#d1d5db', strokeWidth: 1 }}
                        tickFormatter={(value) => `${parseFloat(value).toFixed(2)}%`}
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
                          fontSize: '12px'
                        }}
                        formatter={(value, name) => [`${parseFloat(value).toFixed(2)}%`, 'P&L']}
                        labelFormatter={(label) => `Trade ${parseInt(label) + 1}`}
                      />
                      <Bar 
                        dataKey="pnl" 
                        fill={(entry) => {
                          const pnl = entry.pnl;
                          if (pnl > 0) return '#10b981';
                          if (pnl < 0) return '#ef4444';
                          return '#6b7280';
                        }}
                        radius={[3, 3, 0, 0]}
                        stroke={(entry) => {
                          const pnl = entry.pnl;
                          if (pnl > 0) return '#059669';
                          if (pnl < 0) return '#dc2626';
                          return '#4b5563';
                        }}
                        strokeWidth={1}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Donut Chart */}
              <div className="flex flex-col h-full focus:outline-none" tabIndex="-1" style={{ outline: 'none' }}>
                <h4 className="text-lg font-semibold text-gray-900 mb-3 text-center">Win/Loss Summary</h4>
                <div className="flex-1 flex flex-col justify-start items-center focus:outline-none" tabIndex="-1" style={{ outline: 'none' }}>
                  <div className="w-full max-w-xs mb-2 focus:outline-none" tabIndex="-1" style={{ outline: 'none' }}>
                    <ResponsiveContainer width="100%" height={220} className="focus:outline-none" style={{ outline: 'none' }}>
                      <PieChart className="focus:outline-none" style={{ outline: 'none' }}>
                        <defs>
                          <linearGradient id="winGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity={0.8}/>
                            <stop offset="100%" stopColor="#059669" stopOpacity={0.6}/>
                          </linearGradient>
                          <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.8}/>
                            <stop offset="100%" stopColor="#dc2626" stopOpacity={0.6}/>
                          </linearGradient>
                        </defs>
                        
                        <Pie
                          data={[
                            { name: 'Wins', value: winLossData?.wins?.percentage || 0, color: 'url(#winGradient)', strokeColor: '#10b981' },
                            { name: 'Losses', value: winLossData?.losses?.percentage || 0, color: 'url(#lossGradient)', strokeColor: '#ef4444' }
                          ]}
                          cx="50%"
                          cy="50%"
                          innerRadius={45}
                          outerRadius={90}
                          paddingAngle={8}
                          dataKey="value"
                          stroke="#ffffff"
                          strokeWidth={2}
                        >
                          {[
                            { name: 'Wins', value: winLossData?.wins?.percentage || 0, color: 'url(#winGradient)', strokeColor: '#10b981' },
                            { name: 'Losses', value: winLossData?.losses?.percentage || 0, color: 'url(#lossGradient)', strokeColor: '#ef4444' }
                          ].map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} stroke={entry.strokeColor} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: '1px solid #e5e7eb',
                            borderRadius: '8px',
                            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
                            fontSize: '12px'
                          }}
                          formatter={(value) => [`${parseFloat(value).toFixed(2)}%`, 'Percentage']}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  
                  {/* Stats Cards Inside Win/Loss Analysis */}
                  <div className="w-full grid grid-cols-3 gap-3 -mt-2">
                    {/* Wins Card */}
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg shadow-sm p-3 border border-green-100">
                      <div className="flex items-center space-x-2 mb-1">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-xs font-semibold text-gray-900">Wins</span>
                      </div>
                      <div className="text-lg font-bold text-green-600">
                        {parseFloat(winLossData?.wins?.percentage || 0).toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-600">
                        {winLossData?.wins?.count || 0} trades
                      </div>
                    </div>
                    
                    {/* Losses Card */}
                    <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-lg shadow-sm p-3 border border-red-100">
                      <div className="flex items-center space-x-2 mb-1">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        <span className="text-xs font-semibold text-gray-900">Losses</span>
                      </div>
                      <div className="text-lg font-bold text-red-600">
                        {parseFloat(winLossData?.losses?.percentage || 0).toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-600">
                        {winLossData?.losses?.count || 0} trades
                      </div>
                    </div>
                    
                    {/* Total Trades Card */}
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow-sm p-3 border border-blue-100">
                      <div className="flex items-center space-x-2 mb-1">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <span className="text-xs font-semibold text-gray-900">Total</span>
                      </div>
                      <div className="text-lg font-bold text-blue-600">
                        {winLossData?.total || 0}
                      </div>
                      <div className="text-xs text-gray-600">
                        trades
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-80">
              <div className="text-center">
                <div className="w-20 h-20 bg-gradient-to-br from-gray-100 to-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-10 h-10 text-gray-400" />
                </div>
                <h4 className="text-lg font-semibold text-gray-700 mb-2">No Win/Loss Data</h4>
                <p className="text-gray-500 text-sm max-w-md">
                  Win/Loss data will appear here once TP and SL trades are available for this strategy
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Comprehensive Evaluatory Metrics Section */}
        <div className="bg-gradient-to-br from-white via-gray-50 to-blue-50 rounded-2xl shadow-2xl p-8 mt-8 border border-gray-100">
          <div className="flex items-center space-x-4 mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-1">Comprehensive Evaluatory Metrics</h2>
              <p className="text-gray-600">Advanced performance and risk analysis</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Performance Metrics */}
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-all duration-300">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Performance Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Return</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.totalReturn || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Daily Return</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.dailyReturn || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Weekly Return</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.weeklyReturn || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Monthly Return</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.monthlyReturn || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">CAGR</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.cagr || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Sharpe Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.sharpeRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Sortino Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.sortinoRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Calmar Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.calmarRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Alpha</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.alpha || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Beta</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.beta || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">R²</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.r2 || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Information Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.informationRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Treynor Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.treynorRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Profit Factor</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.profitFactor || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Omega Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.omegaRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Gain to Pain Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.gainToPainRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Payoff Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.payoffRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">CPC Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.cpcRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Risk Return Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.riskReturnRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Common Sense Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.performance?.commonSenseRatio || 0).toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Risk Metrics */}
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-all duration-300">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Risk Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Max Drawdown</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.maxDrawdown || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Max Drawdown Days</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.maxDrawdownDays || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Drawdown</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.avgDrawdown || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Drawdown Days</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.avgDrawdownDays || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Current Drawdown</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.currentDrawdown || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Current Drawdown Days</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.currentDrawdownDays || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Drawdown Duration</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.drawdownDuration || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Conditional Drawdown at Risk</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.conditionalDrawdownAtRisk || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Ulcer Index</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.ulcerIndex || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Risk of Ruin</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.riskOfRuin || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">VaR (95%)</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.var_95 || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">CVaR (95%)</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.cvar_95 || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Downside Deviation</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.downsideDeviation || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Volatility</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.volatility || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Annualized Volatility</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.risk?.annualizedVolatility || 0).toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Trade Metrics */}
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-all duration-300">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Trade Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Number of Trades</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.numberOfTrades || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Win Rate</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.winRate || 0).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Loss Rate</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.lossRate || 0).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Win</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.averageWin || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Loss</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.averageLoss || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Trade Duration</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.averageTradeDuration || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Largest Win</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.largestWin || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Largest Loss</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.largestLoss || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Consecutive Wins</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.consecutiveWins || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Consecutive Losses</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.consecutiveLosses || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Trade Return</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.avgTradeReturn || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Profitability per Trade</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.profitabilityPerTrade || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Common Sense Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.commonSenseRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Recovery Factor</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.trade?.recoveryFactor || 0).toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Profitability Metrics */}
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-all duration-300">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Profitability Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Profit</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.totalProfit || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Loss</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.totalLoss || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Net Profit</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.netProfit || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Risk Return Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.riskReturnRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Common Sense Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.commonSenseRatio || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Conditional Drawdown at Risk</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.conditionalDrawdownAtRisk || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Profit per Trade</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.avgProfitPerTrade || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Loss per Trade</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.avgLossPerTrade || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Profit Loss Ratio</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.profitability?.profitLossRatio || 0).toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Statistical Metrics */}
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-all duration-300">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Statistical Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Skewness</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.statistical?.skewness || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Kurtosis</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.statistical?.kurtosis || 0).toFixed(2)}</span>
                  </div>
              </div>
            </div>

            {/* Monthly and Weekly Metrics */}
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-all duration-300">
              <h4 className="font-semibold text-gray-900 mb-3 text-sm">Monthly & Weekly Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Winning Weeks</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.monthlyWeekly?.winningWeeks || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Losing Weeks</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.monthlyWeekly?.losingWeeks || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Winning Months</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.monthlyWeekly?.winningMonths || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Losing Months</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.monthlyWeekly?.losingMonths || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Positive Months (%)</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.monthlyWeekly?.positiveMonthsPercent || 0).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Negative Months (%)</span>
                  <span className="font-medium text-gray-900">{parseFloat(metricsData?.monthlyWeekly?.negativeMonthsPercent || 0).toFixed(2)}%</span>
                  </div>
              </div>
            </div>
          </div>
        </div>

        {/* Strategy Ledger Section */}
        <div className="bg-gradient-to-br from-white via-gray-50 to-blue-50 rounded-2xl shadow-2xl p-8 mt-8 border border-gray-100">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center space-x-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-1">Strategy Ledger</h2>
                <p className="text-gray-600">Complete trade history with professional analytics</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 bg-white rounded-lg px-4 py-2 shadow-sm border border-gray-200">
                <span className="text-sm font-medium text-gray-600">Rows per page:</span>
                <select
                  className="border-none bg-transparent text-sm font-semibold text-blue-600 focus:outline-none focus:ring-0"
                  value={itemsPerPage}
                  onChange={(e) => {
                    const newItemsPerPage = parseInt(e.target.value);
                    table.setPageSize(newItemsPerPage);
                    setItemsPerPage(newItemsPerPage);
                  }}
                >
                  {[10, 25, 50, 100, 250, 500].map(pageSize => (
                    <option key={pageSize} value={pageSize}>
                      {pageSize}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {ledgerLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              <span className="ml-3 text-gray-600">Loading trade history...</span>
            </div>
          ) : ledgerData && ledgerData.length > 0 ? (
            <>
              {/* TanStack Table Implementation */}
              <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead className="bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50">
                      {table.getHeaderGroups().map(headerGroup => (
                        <tr key={headerGroup.id}>
                          {headerGroup.headers.map(header => (
                            <th
                              key={header.id}
                              className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider border-b border-gray-200 bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50"
                              style={{ width: header.getSize() }}
                            >
                              {header.isPlaceholder
                                ? null
                                : flexRender(
                                    header.column.columnDef.header,
                                    header.getContext()
                                  )}
                            </th>
                          ))}
                        </tr>
                      ))}
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-100">
                        {table.getRowModel().rows.map(row => (
                          <tr 
                            key={row.id} 
                            className="hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 transition-all duration-300 transform hover:scale-[1.01] hover:shadow-md"
                          >
                            {row.getVisibleCells().map(cell => (
                              <td
                                key={cell.id}
                                className="px-6 py-4 whitespace-nowrap text-sm"
                                style={{ width: cell.column.getSize() }}
                              >
                                {flexRender(cell.column.columnDef.cell, cell.getContext())}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                  </table>
                </div>
              </div>

              {/* Enhanced Professional Pagination */}
              <div className="flex items-center justify-between mt-8 bg-gradient-to-r from-white to-gray-50 rounded-xl shadow-lg border border-gray-200 p-6">
                <div className="flex items-center space-x-4">
                  <button
                    type="button"
                    onClick={() => table.previousPage()}
                    disabled={!table.getCanPreviousPage()}
                    className={`flex items-center px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${
                      table.getCanPreviousPage()
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg hover:shadow-xl hover:scale-105'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                    }`}
                  >
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Previous
                  </button>
                  
                  {/* Page Numbers */}
                  <div className="flex items-center space-x-2">
                    {/* First page */}
                    <button
                      type="button"
                      onClick={() => table.setPageIndex(0)}
                      className={`px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                        1 === table.getState().pagination.pageIndex + 1
                          ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                          : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300 shadow-sm hover:shadow-md'
                      }`}
                    >
                      1
                    </button>
                    
                    {/* Middle pages */}
                    {Array.from({ length: Math.min(3, table.getPageCount() - 2) }, (_, i) => {
                      const pageNum = i + 2;
                      const currentPage = table.getState().pagination.pageIndex + 1;
                      
                      if (pageNum <= table.getPageCount() - 1) {
                        return (
                          <button
                            key={pageNum}
                            type="button"
                            onClick={() => table.setPageIndex(pageNum - 1)}
                            className={`px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                              pageNum === currentPage
                                ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                                : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300 shadow-sm hover:shadow-md'
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      }
                      return null;
                    })}
                    
                    {/* Ellipsis if there are more pages */}
                    {table.getPageCount() > 4 && (
                      <span className="px-2 text-gray-500">...</span>
                    )}
                    
                    {/* Last page */}
                    {table.getPageCount() > 1 && (
                      <button
                        type="button"
                        onClick={() => table.setPageIndex(table.getPageCount() - 1)}
                        className={`px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                          table.getPageCount() === table.getState().pagination.pageIndex + 1
                            ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                            : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300 shadow-sm hover:shadow-md'
                        }`}
                      >
                        {table.getPageCount()}
                      </button>
                    )}
                  </div>
                  
                  <button
                    type="button"
                    onClick={() => table.nextPage()}
                    disabled={!table.getCanNextPage()}
                    className={`flex items-center px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${
                      table.getCanNextPage()
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg hover:shadow-xl hover:scale-105'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                    }`}
                  >
                    Next
                    <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
                
                {/* Page Info */}
                <div className="flex items-center space-x-2 bg-white rounded-lg px-4 py-2 shadow-sm border border-gray-200">
                  <span className="text-sm font-medium text-gray-600">Page</span>
                  <span className="text-lg font-bold text-blue-600">{table.getState().pagination.pageIndex + 1}</span>
                  <span className="text-sm text-gray-600">of</span>
                  <span className="text-lg font-bold text-gray-900">{table.getPageCount()}</span>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-16">
              <div className="w-20 h-20 bg-gradient-to-br from-gray-100 to-gray-200 rounded-full flex items-center justify-center mx-auto mb-6">
                <BarChart3 className="w-10 h-10 text-gray-400" />
              </div>
              <h4 className="text-lg font-semibold text-gray-700 mb-2">No Trade History</h4>
              <p className="text-gray-500 text-sm max-w-md mx-auto">
                Trade history will appear here once transactions are available for this strategy
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default StrategyDetail; 