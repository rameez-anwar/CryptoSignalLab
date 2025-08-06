import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { ArrowLeft, TrendingUp, TrendingDown, BarChart3, AlertCircle } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

function StrategyDetail({ strategyName, onBack }) {
  const [strategy, setStrategy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pnlSeries, setPnlSeries] = useState([]);

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

  useEffect(() => {
    fetchStrategyDetails();
    fetchPnlData();
  }, [fetchStrategyDetails, fetchPnlData]);

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
          <div className="flex flex-col justify-between min-w-[200px] max-w-[220px] flex-shrink-0 border-r border-gray-100 pr-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <h2 className="text-xl font-bold text-gray-900">{strategy.name}</h2>
            </div>
            <div className="flex flex-wrap gap-2 mb-4">
              <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                {strategy.parameters.general.symbol?.toUpperCase()}
              </span>
              <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm font-medium">
                {strategy.parameters.general.time_horizon}
              </span>
              <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                {strategy.parameters.general.data_exchange?.toUpperCase()}
              </span>
              {strategy.parameters.live.take_profit && (
                <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                  TP: {(strategy.parameters.live.take_profit * 100).toFixed(2)}%
                </span>
              )}
              {strategy.parameters.live.stop_loss && (
                <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium">
                  SL: {(strategy.parameters.live.stop_loss * 100).toFixed(2)}%
                </span>
              )}
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center bg-gradient-to-r from-green-50 to-green-100 rounded-lg px-3 py-2">
                <span className="text-xs font-medium text-gray-700">Total Return</span>
                <span className="text-lg font-bold text-green-600">{strategy.performance.totalReturn.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between items-center bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg px-3 py-2">
                <span className="text-xs font-medium text-gray-700">Total Trades</span>
                <span className="text-lg font-bold text-blue-600">{strategy.performance.totalTrades}</span>
              </div>
            </div>
          </div>

          {/* Center: Historical Performance */}
          <div className="flex-1 flex flex-col justify-center items-center px-4 border-r border-gray-100">
            <h3 className="text-sm font-semibold text-gray-900 mb-2 text-center">Historical Performance</h3>
            <div className="grid grid-cols-3 gap-3 w-full max-w-md">
              {Object.entries(strategy.performance.historicalReturns).map(([period, value]) => (
                <div key={period} className="bg-gray-50 rounded-lg p-3 text-center border border-gray-200">
                  <div className="text-xs text-gray-500 mb-1">{period} Return</div>
                  <div className={`text-base font-bold ${getPerformanceColor(value)}`}>{value.toFixed(2)}%</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Forecast & Trade Info */}
          <div className="flex flex-col justify-between min-w-[220px] max-w-[260px] flex-shrink-0 pl-6">
            <h3 className="text-sm font-semibold text-gray-900 mb-2">Current Status</h3>
            <div className="grid grid-cols-2 gap-2 mb-2">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Forecast</div>
                <div className="text-base font-semibold text-gray-900">{strategy.forecast.forecast}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Forecast Time</div>
                <div className="text-base font-semibold text-gray-900">{strategy.forecast.forecastTime}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Next Forecast</div>
                <div className="text-base font-semibold text-gray-900">{strategy.forecast.nextForecast}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Entry Price</div>
                <div className="text-base font-semibold text-gray-900">{strategy.performance.entryPrice.toFixed(5)}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Current Price</div>
                <div className="text-base font-semibold text-gray-900">{strategy.performance.currentPrice.toFixed(5)}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Current PNL</div>
                <div className={`text-base font-semibold ${getPerformanceColor(strategy.performance.currentPnl)}`}>{strategy.performance.currentPnl.toFixed(2)}%</div>
              </div>
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
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Cumulative P&L Performance</h3>
              <p className="text-gray-600 text-sm">Real-time profit and loss tracking over time</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full"></div>
                <span className="text-sm font-medium text-gray-700">P&L Trend</span>
              </div>
              <div className="text-right">
                <div className="text-lg font-bold text-gray-900">
                  {pnlSeries.length > 0 ? `${pnlSeries[pnlSeries.length - 1]?.pnl?.toFixed(2) || '0.00'}%` : '0.00%'}
                </div>
                <div className="text-xs text-gray-500">Current P&L</div>
              </div>
            </div>
          </div>
          
          {pnlSeries.length > 0 ? (
            <div className="relative">
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart 
                  data={pnlSeries} 
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
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
                    tickFormatter={(value) => `${value.toFixed(1)}%`}
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
      </main>
    </div>
  );
}

export default StrategyDetail; 