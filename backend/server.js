const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const { Pool } = require('pg');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Database connection
const pool = new Pool({
  user: process.env.PG_USER || 'postgres',
  host: process.env.PG_HOST || 'localhost',
  database: process.env.PG_DB || 'crypto_signals',
  password: process.env.PG_PASSWORD || 'your_password_here',
  port: process.env.PG_PORT || 5432,
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'Crypto Signal Lab API',
    version: '1.0.0',
    endpoints: {
      strategies: '/api/strategies'
    }
  });
});

// Get all simulator strategies from database
app.get('/api/strategies', async (req, res) => {
  try {
    const query = `
      SELECT 
        name,
        exchange,
        symbol,
        time_horizon,
        take_profit,
        stop_loss
      FROM public.config_strategies 
      ORDER BY name ASC
    `;
    
    const result = await pool.query(query);
    
    // Transform database results and fetch P&L from backtest tables
    const strategies = await Promise.all(result.rows.map(async (row, index) => {
      // Try to get P&L from strategies_backtest schema
      let pnl = 0;
      try {
        const backtestQuery = `
          SELECT pnl_sum 
          FROM strategies_backtest.${row.name}_backtest
          ORDER BY ctid DESC
          LIMIT 1
        `;
        const backtestResult = await pool.query(backtestQuery);
        if (backtestResult.rows.length > 0 && backtestResult.rows[0].pnl_sum !== null) {
          pnl = parseFloat(backtestResult.rows[0].pnl_sum);
        }
      } catch (backtestError) {
        // If backtest table doesn't exist or has no data, use 0
        console.log(`No backtest data for ${row.name}_backtest: ${backtestError.message}`);
        pnl = 0;
      }
      
      return {
        id: index + 1,
        name: row.name,
        parameters: {
          exchange: row.exchange,
          symbol: row.symbol,
          timeframe: row.time_horizon,
          takeProfit: row.take_profit,
          stopLoss: row.stop_loss
        },
        performance: {
          profitLoss: pnl
        },
        status: "Active",
        lastUpdated: new Date().toISOString()
      };
    }));

    res.json({
      success: true,
      data: strategies,
      count: strategies.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch strategies from database',
      message: error.message
    });
  }
});

// Get strategy by ID
app.get('/api/strategies/:id', async (req, res) => {
  try {
    const query = `
      SELECT * FROM public.config_strategies 
      WHERE name = $1
    `;
    
    const result = await pool.query(query, [req.params.id]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Strategy not found'
      });
    }
    
    const row = result.rows[0];
    
    // Try to get P&L from strategies_backtest schema
    let pnl = 0;
    try {
      const backtestQuery = `
        SELECT pnl_sum 
        FROM strategies_backtest.${row.name}_backtest
        ORDER BY ctid DESC
        LIMIT 1
      `;
      const backtestResult = await pool.query(backtestQuery);
      if (backtestResult.rows.length > 0 && backtestResult.rows[0].pnl_sum !== null) {
        pnl = parseFloat(backtestResult.rows[0].pnl_sum);
      }
    } catch (backtestError) {
      console.log(`No backtest data for ${row.name}_backtest: ${backtestError.message}`);
      pnl = 0;
    }
    
    const strategy = {
      id: 1,
      name: row.name,
      parameters: {
        exchange: row.exchange,
        symbol: row.symbol,
        timeframe: row.time_horizon,
        takeProfit: row.take_profit,
        stopLoss: row.stop_loss
      },
      performance: {
        profitLoss: pnl
      },
      status: "Active",
      lastUpdated: new Date().toISOString()
    };
    
    res.json({
      success: true,
      data: strategy
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch strategy',
      message: error.message
    });
  }
});

// Get detailed strategy information
app.get('/api/strategies/:id/details', async (req, res) => {
  try {
    const strategyName = req.params.id;
    // 1. Fetch all config parameters
    const basicQuery = `SELECT * FROM public.config_strategies WHERE name = $1`;
    const basicResult = await pool.query(basicQuery, [strategyName]);
    if (basicResult.rows.length === 0) {
      return res.status(404).json({ success: false, error: 'Strategy not found' });
    }
    const row = basicResult.rows[0];

    // 2. Calculate total return (sum of pnl_percent)
    let totalReturn = 0;
    try {
      const totalReturnQuery = `SELECT SUM(pnl_percent) as total_return FROM strategies_backtest.${strategyName}_backtest`;
      const totalReturnResult = await pool.query(totalReturnQuery);
      totalReturn = parseFloat(totalReturnResult.rows[0].total_return || 0);
    } catch (e) { totalReturn = 0; }

    // 3. Calculate total trades (count of buy/sell actions)
    let totalTrades = 0;
    try {
      const totalTradesQuery = `SELECT COUNT(*) as total_trades FROM strategies_backtest.${strategyName}_backtest WHERE action IN ('buy', 'sell')`;
      const totalTradesResult = await pool.query(totalTradesQuery);
      totalTrades = parseInt(totalTradesResult.rows[0].total_trades || 0);
    } catch (e) { totalTrades = 0; }

    // 4. Entry Price (last buy) and Current Price (last sell or last price)
    let entryPrice = 0, currentPrice = 0;
    try {
      const priceQuery = `SELECT buy_price, sell_price, action FROM strategies_backtest.${strategyName}_backtest WHERE buy_price > 0 OR sell_price > 0 ORDER BY datetime DESC LIMIT 20`;
      const priceResult = await pool.query(priceQuery);
      const lastBuy = priceResult.rows.find(r => r.action === 'buy' && r.buy_price > 0);
      const lastSell = priceResult.rows.find(r => r.action === 'sell' && r.sell_price > 0);
      if (lastBuy) entryPrice = parseFloat(lastBuy.buy_price);
      if (lastSell) currentPrice = parseFloat(lastSell.sell_price);
      if (!currentPrice && priceResult.rows.length > 0) {
        const last = priceResult.rows[0];
        if (last.buy_price > 0) currentPrice = parseFloat(last.buy_price);
      }
    } catch (e) {}

    // 5. Current PnL
    let currentPnl = 0;
    if (entryPrice > 0 && currentPrice > 0) {
      currentPnl = ((currentPrice - entryPrice) / entryPrice) * 100;
    }

    // 6. Historical Returns (sum of pnl_percent for each period)
    const periods = [1, 7, 15, 30, 45, 60];
    let historicalReturns = {};
    for (const d of periods) {
      try {
        const q = `SELECT SUM(pnl_percent) as ret FROM strategies_backtest.${strategyName}_backtest WHERE datetime >= NOW() - INTERVAL '${d} days'`;
        const r = await pool.query(q);
        historicalReturns[`${d}d`] = parseFloat(r.rows[0].ret || 0);
      } catch (e) { historicalReturns[`${d}d`] = 0; }
    }

    // 7. Group parameters as in the image
    const general = {
      name: row.name,
      symbol: row.symbol,
      time_horizon: row.time_horizon,
      data_exchange: row.exchange,
      use_ml: row.use_ml,
      use_pattern: row.use_pattern,
      use_filters: row.use_filters,
      is_live: row.is_live,
      simulator: row.simulator,
      execution: row.execution,
      priority: row.priority,
      use_fib_levels: row.use_fib_levels,
      use_head_shoulders: row.use_head_shoulders,
      use_cup_handle: row.use_cup_handle
    };
    // List of main indicator columns to show in the Filters card (with unwanted ones removed)
    const filterIndicators = [
      'rsi', 'stoch_k', 'stoch_d', 'bb_lower', 'bb_middle', 'bb_upper',
      'sma', 'ema', 'macd', 'psar', 'adx'
    ];
    // Helper to convert indicator names to snake_case
    function toSnakeCase(str) {
      return str.replace(/([a-z])([A-Z])/g, '$1_$2').replace(/\s+/g, '_').toLowerCase();
    }
    const filters = {};
    filterIndicators.forEach(ind => {
      filters[`filters_use_${toSnakeCase(ind)}`] = !!row[ind];
    });
    if (row.filters_percent_required !== null && row.filters_percent_required !== undefined) {
      filters.percent_required = row.filters_percent_required;
    }
    const patterns = {
      patterns_use_complete_patterns: row.patterns_use_complete_patterns,
      patterns_use_incomplete_patterns: row.patterns_use_incomplete_patterns,
      patterns_list_complete_patterns: row.patterns_list_complete_patterns,
      patterns_list_incomplete_patterns: row.patterns_list_incomplete_patterns,
      patterns_window: row.patterns_window,
      patterns_sl_ratio: row.patterns_sl_ratio,
      patterns_tp1_ratio: row.patterns_tp1_ratio,
      patterns_tp2_ratio: row.patterns_tp2_ratio,
      patterns_allowed_error: row.patterns_allowed_error,
      patterns_tie_breaker: row.patterns_tie_breaker
    };
    
    // Add all pattern recognition columns from database
    const patternList = ['cdldoji', 'cdlhammer', 'cdlengulfing', 'cdlmorningstar', 'cdleveningstar', 'cdlshootingstar', 'cdlhangingman', 'cdldarkcloudcover', 'cdlpiercing', 'cdl3whitesoldiers', 'cdl3blackcrows'];
    
    // Add pattern recognition columns to patterns object
    patternList.forEach(pattern => {
      if (row.hasOwnProperty(pattern)) {
        patterns[pattern] = !!row[pattern];
      }
    });
    
    // Dynamically create indicator cards based on enabled indicators
    const indicators = {};
    
    // Get all column names from the row
    const columnNames = Object.keys(row);
    
    // Find all indicator columns and their window sizes
    const indicatorData = {};
    columnNames.forEach(col => {
      if (col.endsWith('_window_size') && row[col] !== null && row[col] !== undefined) {
        const indicatorName = col.replace('_window_size', '');
        const isEnabled = row[indicatorName] === true;
        const windowSize = row[col];
        
        if (isEnabled && windowSize > 0) {
          indicatorData[indicatorName] = windowSize;
        }
      }
    });
    
    // Group indicators by category
    const overlapStudies = ['sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'mama', 't3', 'midpoint', 'midprice', 'bb_upper', 'bb_middle', 'bb_lower', 'parabolic_sar', 'sarext', 'donchian_upper', 'donchian_lower', 'ht_trendline'];
    const momentumIndicators = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'adx', 'adxr', 'cci', 'willr', 'roc', 'trix', 'stoch_k', 'stoch_d', 'stochf_k', 'stochf_d', 'stochrsi_k', 'stochrsi_d', 'ultosc', 'cmo', 'apo', 'ppo', 'mom', 'dx', 'macdext', 'macdext_signal', 'macdext_hist', 'macdfix', 'macdfix_signal', 'macdfix_hist', 'plus_di', 'plus_dm', 'minus_di', 'minus_dm', 'rocp', 'rocr', 'rocr100', 'aroon_down', 'aroon_up', 'aroon_osc', 'bop'];
    const volumeIndicators = ['obv', 'mfi', 'ad', 'adosc'];
    const volatilityIndicators = ['atr', 'natr', 'trange', 'chaikin_volatility'];
    const priceTransform = ['avgprice', 'medprice', 'typprice', 'wclprice'];
    const cycleIndicators = ['ht_dcperiod', 'ht_dcphase', 'ht_phasor', 'ht_sine', 'ht_trendmode'];
    const patternRecognition = ['cdl2crows', 'cdl3blackcrows', 'cdl3inside', 'cdl3linestrike', 'cdl3outside', 'cdl3starsinsouth', 'cdl3whitesoldiers', 'cdlabandonedbaby', 'cdladvanceblock', 'cdlbelthold', 'cdlbreakaway', 'cdlclosingmarubozu', 'cdlconcealbabyswall', 'cdlcounterattack', 'cdldarkcloudcover', 'cdldoji', 'cdldojistar', 'cdldragonflydoji', 'cdlengulfing', 'cdleveningdojistar', 'cdleveningstar', 'cdlgapsidesidewhite', 'cdlgravestonedoji', 'cdlhammer', 'cdlhangingman', 'cdlharami', 'cdlharamicross', 'cdlhighwave', 'cdlhikkake', 'cdlhikkakemod', 'cdlhomingpigeon', 'cdlidentical3crows', 'cdlinneck', 'cdlinvertedhammer', 'cdlkicking', 'cdlkickingbylength', 'cdlladderbottom', 'cdllongleggeddoji', 'cdllongline', 'cdlmarubozu', 'cdlmatchinglow', 'cdlmathold', 'cdlmorningdojistar', 'cdlmorningstar', 'cdlonneck', 'cdlpiercing', 'cdlrickshawman', 'cdlrisefall3methods', 'cdlseparatinglines', 'cdlshootingstar', 'cdlshortline', 'cdlspinningtop', 'cdlstalledpattern', 'cdlsticksandwich', 'cdltakuri', 'cdltasukigap', 'cdlthrusting', 'cdltristar', 'cdlunique3river', 'cdlupsidegap2crows', 'cdlxsidgap3methods'];
    
    // Create indicator objects for each category
    const overlapIndicators = {};
    const momentumIndicatorsObj = {};
    const volumeIndicatorsObj = {};
    const volatilityIndicatorsObj = {};
    const priceTransformObj = {};
    const cycleIndicatorsObj = {};
    const patternRecognitionObj = {};
    
    Object.entries(indicatorData).forEach(([indicator, windowSize]) => {
      if (overlapStudies.includes(indicator)) {
        overlapIndicators[indicator] = windowSize;
      } else if (momentumIndicators.includes(indicator)) {
        momentumIndicatorsObj[indicator] = windowSize;
      } else if (volumeIndicators.includes(indicator)) {
        volumeIndicatorsObj[indicator] = windowSize;
      } else if (volatilityIndicators.includes(indicator)) {
        volatilityIndicatorsObj[indicator] = windowSize;
      } else if (priceTransform.includes(indicator)) {
        priceTransformObj[indicator] = windowSize;
      } else if (cycleIndicators.includes(indicator)) {
        cycleIndicatorsObj[indicator] = windowSize;
      } else if (patternRecognition.includes(indicator)) {
        patternRecognitionObj[indicator] = windowSize;
      }
    });
    
    // Only include categories that have enabled indicators
    if (Object.keys(overlapIndicators).length > 0) {
      indicators.overlapStudies = overlapIndicators;
    }
    if (Object.keys(momentumIndicatorsObj).length > 0) {
      indicators.momentum = momentumIndicatorsObj;
    }
    if (Object.keys(volumeIndicatorsObj).length > 0) {
      indicators.volume = volumeIndicatorsObj;
    }
    if (Object.keys(volatilityIndicatorsObj).length > 0) {
      indicators.volatility = volatilityIndicatorsObj;
    }
    if (Object.keys(priceTransformObj).length > 0) {
      indicators.priceTransform = priceTransformObj;
    }
    if (Object.keys(cycleIndicatorsObj).length > 0) {
      indicators.cycle = cycleIndicatorsObj;
    }
    if (Object.keys(patternRecognitionObj).length > 0) {
      indicators.patterns = patternRecognitionObj;
    }
    
    const live = {
      live_tp_sl: row.live_tp_sl,
      live_tp_percent: row.live_tp_percent,
      live_sl_percent: row.live_sl_percent,
      take_profit: row.take_profit,
      stop_loss: row.stop_loss
    };

    // 8. Forecast (placeholder)
    const forecastData = {
      forecast: '-',
      forecastTime: '-',
      nextForecast: '06 Aug 2025 07:00:00'
    };

    // 9. Compose response
    const detailedStrategy = {
      id: 1,
      name: row.name,
      performance: {
        totalReturn,
        totalTrades,
        entryPrice,
        currentPrice,
        currentPnl,
        historicalReturns
      },
      forecast: forecastData,
      parameters: {
        general,
        filters,
        patterns,
        indicators, // Add indicators to the response
        ema: { ema_window: row.ema_window }, // Keep existing ema
        adx: { adx_window: row.adx_window }, // Keep existing adx
        live
      },
      status: 'Active',
      lastUpdated: new Date().toISOString()
    };
    res.json({ success: true, data: detailedStrategy });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch strategy details', message: error.message });
  }
});

// Add after other endpoints
app.get('/api/strategies/:id/pnl_timeseries', async (req, res) => {
  try {
    const strategyName = req.params.id;
    console.log('Fetching PNL timeseries for strategy:', strategyName);
    
    const query = `
      SELECT datetime, pnl_sum
      FROM strategies_backtest.${strategyName}_backtest
      ORDER BY datetime ASC
    `;
    console.log('Query:', query);
    
    const result = await pool.query(query);
    console.log('PNL data rows:', result.rows.length);
    
    const data = result.rows.map(row => ({
      date: row.datetime,
      pnl: row.pnl_sum
    }));
    
    console.log('First few PNL records:', data.slice(0, 5));
    
    res.json({
      success: true,
      data: data
    });
  } catch (error) {
    console.error('Error fetching PNL timeseries:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch PNL timeseries', message: error.message });
  }
});

// Add win/loss endpoint
app.get('/api/strategies/:id/winloss', async (req, res) => {
  try {
    const strategyName = req.params.id;
    console.log('Fetching win/loss data for strategy:', strategyName);
    
    // Get all PNL data for individual bar chart
    const pnlQuery = `
      SELECT pnl_percent
      FROM strategies_backtest.${strategyName}_backtest
      WHERE pnl_percent IS NOT NULL
      ORDER BY datetime ASC
    `;
    
    const pnlResult = await pool.query(pnlQuery);
    console.log('PNL data rows:', pnlResult.rows.length);
    
    // Calculate win/loss based on PNL values
    let winCount = 0;
    let lossCount = 0;
    const individualPnl = [];
    
    pnlResult.rows.forEach(row => {
      const pnl = parseFloat(row.pnl_percent);
      
      // Skip transaction fee (-0.05)
      if (pnl === -0.05) {
        return;
      }
      
      // Add to individual PNL array for bar chart
      individualPnl.push(pnl);
      
      // Count as win or loss
      if (pnl > 0) {
        winCount++;
      } else if (pnl < 0) {
        lossCount++;
      }
    });
    
    const total = winCount + lossCount;
    const winPercentage = total > 0 ? ((winCount / total) * 100).toFixed(1) : 0;
    const lossPercentage = total > 0 ? ((lossCount / total) * 100).toFixed(1) : 0;
    
    const winLossData = {
      wins: {
        count: winCount,
        percentage: parseFloat(winPercentage)
      },
      losses: {
        count: lossCount,
        percentage: parseFloat(lossPercentage)
      },
      total: total,
      individualPnl: individualPnl
    };
    
    console.log('Calculated win/loss data:', winLossData);
    
    res.json({
      success: true,
      data: winLossData
    });
  } catch (error) {
    console.error('Error fetching win/loss data:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch win/loss data', message: error.message });
  }
});

// Add comprehensive metrics endpoint
app.get('/api/strategies/:id/metrics', async (req, res) => {
  try {
    const strategyName = req.params.id;
    console.log('Fetching comprehensive metrics for strategy:', strategyName);
    
    // Get all backtest data
    const query = `
      SELECT * FROM strategies_backtest.${strategyName}_backtest
      ORDER BY datetime ASC
    `;
    
    const result = await pool.query(query);
    console.log('Backtest data rows:', result.rows.length);
    
    if (result.rows.length === 0) {
      return res.json({
        success: true,
        data: {
          performance: {},
          risk: {},
          trade: {},
          profitability: {},
          statistical: {},
          monthlyWeekly: {}
        }
      });
    }
    
    const data = result.rows;
    
    // Calculate Performance Metrics
    const totalReturn = data.reduce((sum, row) => sum + (parseFloat(row.pnl_percent) || 0), 0);
    const dailyReturn = data.length > 0 ? totalReturn / data.length : 0;
    const weeklyReturn = data.filter(row => {
      const date = new Date(row.datetime);
      const now = new Date();
      const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
      return diffDays <= 7;
    }).reduce((sum, row) => sum + (parseFloat(row.pnl_percent) || 0), 0);
    const monthlyReturn = data.filter(row => {
      const date = new Date(row.datetime);
      const now = new Date();
      const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
      return diffDays <= 30;
    }).reduce((sum, row) => sum + (parseFloat(row.pnl_percent) || 0), 0);
    
    // Calculate CAGR (simplified)
    const firstDate = new Date(data[0].datetime);
    const lastDate = new Date(data[data.length - 1].datetime);
    const years = (lastDate - firstDate) / (1000 * 60 * 60 * 24 * 365);
    const cagr = years > 0 ? Math.pow((1 + totalReturn / 100), 1 / years) - 1 : 0;
    
    // Calculate Risk Metrics
    const pnlValues = data.map(row => parseFloat(row.pnl_percent) || 0);
    const maxDrawdown = Math.min(...pnlValues);
    const avgDrawdown = pnlValues.filter(v => v < 0).reduce((sum, v) => sum + v, 0) / pnlValues.filter(v => v < 0).length || 0;
    const volatility = Math.sqrt(pnlValues.reduce((sum, v) => sum + Math.pow(v - (totalReturn / data.length), 2), 0) / data.length);
    
    // Calculate Trade Metrics
    const trades = data.filter(row => row.action && row.action !== 'hold');
    const winTrades = trades.filter(row => parseFloat(row.pnl_percent) > 0);
    const lossTrades = trades.filter(row => parseFloat(row.pnl_percent) < 0);
    const winRate = trades.length > 0 ? (winTrades.length / trades.length) * 100 : 0;
    const lossRate = trades.length > 0 ? (lossTrades.length / trades.length) * 100 : 0;
    
    // Calculate Profitability Metrics
    const totalProfit = winTrades.reduce((sum, row) => sum + (parseFloat(row.pnl_percent) || 0), 0);
    const totalLoss = lossTrades.reduce((sum, row) => sum + (parseFloat(row.pnl_percent) || 0), 0);
    const netProfit = totalProfit + totalLoss;
    const avgProfitPerTrade = winTrades.length > 0 ? totalProfit / winTrades.length : 0;
    const avgLossPerTrade = lossTrades.length > 0 ? totalLoss / lossTrades.length : 0;
    
    // Calculate Statistical Metrics
    const mean = pnlValues.reduce((sum, v) => sum + v, 0) / pnlValues.length;
    const variance = pnlValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / pnlValues.length;
    const skewness = pnlValues.reduce((sum, v) => sum + Math.pow(v - mean, 3), 0) / (pnlValues.length * Math.pow(variance, 1.5));
    const kurtosis = pnlValues.reduce((sum, v) => sum + Math.pow(v - mean, 4), 0) / (pnlValues.length * Math.pow(variance, 2));
    
    // Calculate Monthly/Weekly Metrics
    const now = new Date();
    const weeklyData = data.filter(row => {
      const date = new Date(row.datetime);
      const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
      return diffDays <= 7;
    });
    const monthlyData = data.filter(row => {
      const date = new Date(row.datetime);
      const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
      return diffDays <= 30;
    });
    
    const winningWeeks = weeklyData.filter(row => parseFloat(row.pnl_percent) > 0).length;
    const losingWeeks = weeklyData.filter(row => parseFloat(row.pnl_percent) < 0).length;
    const winningMonths = monthlyData.filter(row => parseFloat(row.pnl_percent) > 0).length;
    const losingMonths = monthlyData.filter(row => parseFloat(row.pnl_percent) < 0).length;
    
    const metrics = {
      performance: {
        totalReturn: parseFloat(totalReturn.toFixed(2)),
        dailyReturn: parseFloat(dailyReturn.toFixed(2)),
        weeklyReturn: parseFloat(weeklyReturn.toFixed(2)),
        monthlyReturn: parseFloat(monthlyReturn.toFixed(2)),
        cagr: parseFloat(cagr.toFixed(2)),
        sharpeRatio: parseFloat((totalReturn / volatility).toFixed(2)),
        sortinoRatio: parseFloat((totalReturn / Math.abs(avgDrawdown)).toFixed(2)),
        calmarRatio: parseFloat((totalReturn / Math.abs(maxDrawdown)).toFixed(2)),
        alpha: parseFloat((totalReturn - (volatility * 0.02)).toFixed(2)),
        beta: parseFloat((volatility / 0.02).toFixed(2)),
        r2: parseFloat((Math.pow(totalReturn, 2) / (Math.pow(totalReturn, 2) + Math.pow(volatility, 2))).toFixed(2)),
        informationRatio: parseFloat((totalReturn / volatility).toFixed(2)),
        treynorRatio: parseFloat((totalReturn / (volatility * 0.02)).toFixed(2)),
        profitFactor: parseFloat((Math.abs(totalProfit) / Math.abs(totalLoss)).toFixed(2)),
        omegaRatio: parseFloat((totalProfit / Math.abs(totalLoss)).toFixed(2)),
        gainToPainRatio: parseFloat((totalProfit / Math.abs(totalLoss)).toFixed(2)),
        payoffRatio: parseFloat((avgProfitPerTrade / Math.abs(avgLossPerTrade)).toFixed(2)),
        cpcRatio: parseFloat((totalReturn / trades.length).toFixed(2)),
        riskReturnRatio: parseFloat((totalReturn / Math.abs(maxDrawdown)).toFixed(2)),
        commonSenseRatio: parseFloat((totalReturn / Math.abs(maxDrawdown)).toFixed(2))
      },
      risk: {
        maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
        maxDrawdownDays: Math.abs(maxDrawdown) > 0 ? Math.ceil(Math.abs(maxDrawdown) / 0.1) : 0,
        avgDrawdown: parseFloat(avgDrawdown.toFixed(2)),
        avgDrawdownDays: Math.abs(avgDrawdown) > 0 ? Math.ceil(Math.abs(avgDrawdown) / 0.1) : 0,
        currentDrawdown: parseFloat(maxDrawdown.toFixed(2)),
        currentDrawdownDays: Math.abs(maxDrawdown) > 0 ? Math.ceil(Math.abs(maxDrawdown) / 0.1) : 0,
        drawdownDuration: Math.abs(maxDrawdown) > 0 ? Math.ceil(Math.abs(maxDrawdown) / 0.1) : 0,
        conditionalDrawdownAtRisk: parseFloat((maxDrawdown * 0.95).toFixed(2)),
        ulcerIndex: parseFloat(Math.sqrt(pnlValues.filter(v => v < 0).reduce((sum, v) => sum + Math.pow(v, 2), 0) / pnlValues.length).toFixed(2)),
        riskOfRuin: parseFloat((Math.pow(Math.abs(avgLossPerTrade) / avgProfitPerTrade, winTrades.length)).toFixed(2)),
        var_95: parseFloat((mean - (1.645 * Math.sqrt(variance))).toFixed(2)),
        cvar_95: parseFloat((mean - (2.326 * Math.sqrt(variance))).toFixed(2)),
        downsideDeviation: parseFloat(Math.sqrt(pnlValues.filter(v => v < 0).reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / pnlValues.length).toFixed(2)),
        volatility: parseFloat(volatility.toFixed(2)),
        annualizedVolatility: parseFloat((volatility * Math.sqrt(365)).toFixed(2))
      },
      trade: {
        numberOfTrades: trades.length,
        winRate: parseFloat(winRate.toFixed(1)),
        lossRate: parseFloat(lossRate.toFixed(1)),
        averageWin: parseFloat(avgProfitPerTrade.toFixed(2)),
        averageLoss: parseFloat(avgLossPerTrade.toFixed(2)),
        averageTradeDuration: parseFloat((data.length / trades.length).toFixed(1)),
        largestWin: parseFloat(Math.max(...winTrades.map(row => parseFloat(row.pnl_percent) || 0)).toFixed(2)),
        largestLoss: parseFloat(Math.min(...lossTrades.map(row => parseFloat(row.pnl_percent) || 0)).toFixed(2)),
        consecutiveWins: 0, // Would need more complex logic
        consecutiveLosses: 0, // Would need more complex logic
        avgTradeReturn: parseFloat((totalReturn / trades.length).toFixed(2)),
        profitabilityPerTrade: parseFloat((netProfit / trades.length).toFixed(2)),
        commonSenseRatio: parseFloat((totalReturn / Math.abs(maxDrawdown)).toFixed(2)),
        recoveryFactor: parseFloat((totalReturn / Math.abs(maxDrawdown)).toFixed(2))
      },
      profitability: {
        totalProfit: parseFloat(totalProfit.toFixed(2)),
        totalLoss: parseFloat(totalLoss.toFixed(2)),
        netProfit: parseFloat(netProfit.toFixed(2)),
        riskReturnRatio: parseFloat((totalReturn / Math.abs(maxDrawdown)).toFixed(2)),
        commonSenseRatio: parseFloat((totalReturn / Math.abs(maxDrawdown)).toFixed(2)),
        conditionalDrawdownAtRisk: parseFloat((maxDrawdown * 0.95).toFixed(2)),
        avgProfitPerTrade: parseFloat(avgProfitPerTrade.toFixed(2)),
        avgLossPerTrade: parseFloat(avgLossPerTrade.toFixed(2)),
        profitLossRatio: parseFloat((Math.abs(totalProfit) / Math.abs(totalLoss)).toFixed(2))
      },
      statistical: {
        skewness: parseFloat(skewness.toFixed(2)),
        kurtosis: parseFloat(kurtosis.toFixed(2))
      },
      monthlyWeekly: {
        winningWeeks: winningWeeks,
        losingWeeks: losingWeeks,
        winningMonths: winningMonths,
        losingMonths: losingMonths,
        positiveMonthsPercent: parseFloat(((winningMonths / (winningMonths + losingMonths)) * 100).toFixed(1)),
        negativeMonthsPercent: parseFloat(((losingMonths / (winningMonths + losingMonths)) * 100).toFixed(1))
      }
    };
    
    res.json({
      success: true,
      data: metrics
    });
  } catch (error) {
    console.error('Error fetching comprehensive metrics:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch comprehensive metrics', message: error.message });
  }
});

// Add ledger endpoint with pagination
app.get('/api/strategies/:id/ledger', async (req, res) => {
  try {
    const strategyName = req.params.id;
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10000; // Support large limits for virtualized table
    const offset = (page - 1) * limit;
    
    console.log('Fetching ledger data for strategy:', strategyName, 'page:', page, 'limit:', limit);
    
    // Get total count
    const countQuery = `
      SELECT COUNT(*) as total
      FROM strategies_backtest.${strategyName}_backtest
      WHERE action IS NOT NULL AND action != 'hold'
    `;
    
    const countResult = await pool.query(countQuery);
    const total = parseInt(countResult.rows[0].total);
    
    // Get paginated data - using only available columns
    const dataQuery = `
      SELECT 
        ROW_NUMBER() OVER (ORDER BY datetime) as id,
        datetime,
        action,
        buy_price,
        sell_price,
        pnl_percent,
        pnl_sum,
        balance
      FROM strategies_backtest.${strategyName}_backtest
      WHERE action IS NOT NULL AND action != 'hold'
      ORDER BY datetime ASC
      LIMIT $1 OFFSET $2
    `;
    
    const dataResult = await pool.query(dataQuery, [limit, offset]);
    
    const ledgerData = dataResult.rows.map(row => ({
      id: row.id,
      time: new Date(row.datetime).toLocaleString('en-US', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      }),
      action: row.action,
      buyPrice: row.buy_price || 0,
      sellPrice: row.sell_price || 0,
      pnlPercent: parseFloat(row.pnl_percent || 0).toFixed(2),
      pnlSum: parseFloat(row.pnl_sum || 0).toFixed(2),
      balance: parseFloat(row.balance || 0).toFixed(2)
    }));
    
    res.json({
      success: true,
      data: {
        ledger: ledgerData,
        pagination: {
          currentPage: page,
          totalPages: Math.ceil(total / limit),
          totalItems: total,
          itemsPerPage: limit,
          hasNextPage: page < Math.ceil(total / limit),
          hasPrevPage: page > 1
        }
      }
    });
  } catch (error) {
    console.error('Error fetching ledger data:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch ledger data', message: error.message });
  }
});

// Add user management endpoints
app.get('/api/users', async (req, res) => {
  try {
    const query = `
      SELECT 
        id,
        name,
        email,
        strategies,
        created_at,
        updated_at
      FROM users.users 
      ORDER BY created_at DESC
    `;
    
    const result = await pool.query(query);
    
    res.json({
      success: true,
      data: result.rows,
      count: result.rows.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch users from database',
      message: error.message
    });
  }
});

app.post('/api/users', async (req, res) => {
  try {
    const { name, email, password, api_key, api_secret, strategies } = req.body;
    
    // Validate required fields
    if (!name || !email || !password || !api_key || !api_secret || !strategies) {
      return res.status(400).json({
        success: false,
        error: 'All fields are required'
      });
    }
    
    // Check if email already exists
    const existingUser = await pool.query('SELECT id FROM users.users WHERE email = $1', [email]);
    if (existingUser.rows.length > 0) {
      return res.status(400).json({
        success: false,
        error: 'User with this email already exists'
      });
    }
    
    // Hash password (in production, use bcrypt)
    const hashedPassword = password; // For now, store as plain text
    
    const query = `
      INSERT INTO users.users (name, email, password, api_key, api_secret, strategies)
      VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING id, name, email, strategies, created_at
    `;
    
    const result = await pool.query(query, [
      name, 
      email, 
      hashedPassword, 
      api_key, 
      api_secret, 
      JSON.stringify(strategies)
    ]);
    
    res.json({
      success: true,
      data: result.rows[0],
      message: 'User created successfully'
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create user',
      message: error.message
    });
  }
});

app.put('/api/users/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { name, email, password, api_key, api_secret, strategies } = req.body;
    
    // Validate required fields
    if (!name || !email || !api_key || !api_secret || !strategies) {
      return res.status(400).json({
        success: false,
        error: 'All fields are required'
      });
    }
    
    // Check if email already exists for other users
    const existingUser = await pool.query(
      'SELECT id FROM users.users WHERE email = $1 AND id != $2', 
      [email, id]
    );
    if (existingUser.rows.length > 0) {
      return res.status(400).json({
        success: false,
        error: 'User with this email already exists'
      });
    }
    
    let query, params;
    
    if (password) {
      // Update with password
      query = `
        UPDATE users.users 
        SET name = $1, email = $2, password = $3, api_key = $4, api_secret = $5, strategies = $6, updated_at = NOW()
        WHERE id = $7
        RETURNING id, name, email, strategies, updated_at
      `;
      params = [name, email, password, api_key, api_secret, JSON.stringify(strategies), id];
    } else {
      // Update without password
      query = `
        UPDATE users.users 
        SET name = $1, email = $2, api_key = $3, api_secret = $4, strategies = $5, updated_at = NOW()
        WHERE id = $6
        RETURNING id, name, email, strategies, updated_at
      `;
      params = [name, email, api_key, api_secret, JSON.stringify(strategies), id];
    }
    
    const result = await pool.query(query, params);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'User not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0],
      message: 'User updated successfully'
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update user',
      message: error.message
    });
  }
});

app.delete('/api/users/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const query = 'DELETE FROM users.users WHERE id = $1 RETURNING id';
    const result = await pool.query(query, [id]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'User not found'
      });
    }
    
    res.json({
      success: true,
      message: 'User deleted successfully'
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete user',
      message: error.message
    });
  }
});

app.get('/api/users/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const query = `
      SELECT id, name, email, api_key, api_secret, strategies, created_at, updated_at
      FROM users.users 
      WHERE id = $1
    `;
    
    const result = await pool.query(query, [id]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'User not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch user',
      message: error.message
    });
  }
});

// Get all ML models from database
app.get('/api/models', async (req, res) => {
  try {
    const query = `
      SELECT 
        id,
        model_name,
        exchange,
        symbol,
        time_horizon,
        table_name,
        final_pnl
      FROM ml_summary.ml_summary 
      ORDER BY final_pnl DESC
    `;
    
    const result = await pool.query(query);
    
    const models = result.rows.map((row, index) => {
      // Calculate status based on performance
      let status = "Active";
      if (row.final_pnl > 10) {
        status = "Excellent";
      } else if (row.final_pnl > 5) {
        status = "Good";
      } else if (row.final_pnl > 0) {
        status = "Profitable";
      } else if (row.final_pnl > -5) {
        status = "Struggling";
      } else {
        status = "Underperforming";
      }

      return {
        id: row.id,
        model_name: row.model_name,
        exchange: row.exchange,
        symbol: row.symbol,
        time_horizon: row.time_horizon,
        table_name: row.table_name,
        final_pnl: parseFloat(row.final_pnl || 0),
        status: status
      };
    });

    res.json({
      success: true,
      data: models,
      count: models.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch ML models from database',
      message: error.message
    });
  }
});

// Get ML model by table name
app.get('/api/models/:tableName', async (req, res) => {
  try {
    const query = `
      SELECT 
        id,
        model_name,
        exchange,
        symbol,
        time_horizon,
        table_name,
        final_pnl
      FROM ml_summary.ml_summary 
      WHERE table_name = $1
    `;
    
    const result = await pool.query(query, [req.params.tableName]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'ML model not found'
      });
    }
    
    const row = result.rows[0];
    
    const model = {
      id: row.id,
      model_name: row.model_name,
      exchange: row.exchange,
      symbol: row.symbol,
      time_horizon: row.time_horizon,
      table_name: row.table_name,
      final_pnl: parseFloat(row.final_pnl || 0)
    };
    
    res.json({
      success: true,
      data: model
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch ML model',
      message: error.message
    });
  }
});

// Get ML model details
app.get('/api/models/:tableName/details', async (req, res) => {
  try {
    const query = `
      SELECT 
        id,
        model_name,
        exchange,
        symbol,
        time_horizon,
        table_name,
        final_pnl
      FROM ml_summary.ml_summary 
      WHERE table_name = $1
    `;
    
    const result = await pool.query(query, [req.params.tableName]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'ML model not found'
      });
    }
    
    const row = result.rows[0];
    
    const model = {
      id: row.id,
      model_name: row.model_name,
      exchange: row.exchange,
      symbol: row.symbol,
      time_horizon: row.time_horizon,
      table_name: row.table_name,
      final_pnl: parseFloat(row.final_pnl || 0)
    };
    
    res.json({
      success: true,
      data: model
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch ML model details',
      message: error.message
    });
  }
});

// Get ML model PNL timeseries
app.get('/api/models/:tableName/pnl_timeseries', async (req, res) => {
  try {
    const tableName = req.params.tableName;
    const query = `
      SELECT 
        datetime,
        pnl_sum as pnl
      FROM ml_ledger.${tableName}
      ORDER BY datetime ASC
    `;
    
    const result = await pool.query(query);
    
    const pnlData = result.rows.map(row => ({
      date: row.datetime,
      pnl: parseFloat(row.pnl || 0)
    }));
    
    res.json({
      success: true,
      data: pnlData
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch PNL timeseries',
      message: error.message
    });
  }
});

// Get ML model win/loss data
app.get('/api/models/:tableName/winloss', async (req, res) => {
  try {
    const tableName = req.params.tableName;
    const query = `
      SELECT 
        pnl_percent
      FROM ml_ledger.${tableName}
      WHERE pnl_percent IS NOT NULL
      ORDER BY datetime ASC
    `;
    
    const result = await pool.query(query);
    
    const pnlValues = result.rows.map(row => parseFloat(row.pnl_percent || 0));
    const totalTrades = pnlValues.length;
    const wins = pnlValues.filter(pnl => pnl > 0).length;
    const losses = pnlValues.filter(pnl => pnl < 0).length;
    
    const winLossData = {
      total: totalTrades,
      wins: {
        count: wins,
        percentage: totalTrades > 0 ? (wins / totalTrades) * 100 : 0
      },
      losses: {
        count: losses,
        percentage: totalTrades > 0 ? (losses / totalTrades) * 100 : 0
      },
      individualPnl: pnlValues
    };
    
    res.json({
      success: true,
      data: winLossData
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch win/loss data',
      message: error.message
    });
  }
});

// Get ML model metrics
app.get('/api/models/:tableName/metrics', async (req, res) => {
  try {
    const tableName = req.params.tableName;
    
    // Get metrics from stats.ml_stats table
    const query = `
      SELECT *
      FROM stats.ml_stats
      WHERE strategy_name = $1
    `;
    
    const result = await pool.query(query, [tableName]);
    
    if (result.rows.length === 0) {
      return res.json({
        success: true,
        data: {
          performance: {},
          risk: {},
          trade: {},
          profitability: {},
          statistical: {},
          monthlyWeekly: {}
        }
      });
    }
    
    const row = result.rows[0];
    
    // Transform the data to match the expected structure
    const metricsData = {
      performance: {
        totalReturn: parseFloat(row.total_return || 0),
        dailyReturn: parseFloat(row.daily_return || 0),
        weeklyReturn: parseFloat(row.weekly_return || 0),
        monthlyReturn: parseFloat(row.monthly_return || 0),
        cagr: parseFloat(row.cagr || 0),
        sharpeRatio: parseFloat(row.sharpe_ratio || 0),
        sortinoRatio: parseFloat(row.sortino_ratio || 0),
        calmarRatio: parseFloat(row.calmar_ratio || 0),
        alpha: parseFloat(row.alpha || 0),
        beta: parseFloat(row.beta || 0),
        r2: parseFloat(row.r2 || 0),
        informationRatio: parseFloat(row.information_ratio || 0),
        treynorRatio: parseFloat(row.treynor_ratio || 0),
        profitFactor: parseFloat(row.profit_factor || 0),
        omegaRatio: parseFloat(row.omega_ratio || 0),
        gainToPainRatio: parseFloat(row.gain_to_pain_ratio || 0),
        payoffRatio: parseFloat(row.payoff_ratio || 0),
        cpcRatio: parseFloat(row.cpc_ratio || 0),
        riskReturnRatio: parseFloat(row.risk_return_ratio || 0),
        commonSenseRatio: parseFloat(row.common_sense_ratio || 0)
      },
      risk: {
        maxDrawdown: parseFloat(row.max_drawdown || 0),
        maxDrawdownDays: parseFloat(row.max_drawdown_days || 0),
        avgDrawdown: parseFloat(row.avg_drawdown || 0),
        avgDrawdownDays: parseFloat(row.avg_drawdown_days || 0),
        currentDrawdown: parseFloat(row.current_drawdown || 0),
        currentDrawdownDays: parseFloat(row.current_drawdown_days || 0),
        drawdownDuration: parseFloat(row.drawdown_duration || 0),
        conditionalDrawdownAtRisk: parseFloat(row.conditional_drawdown_at_risk || 0),
        ulcerIndex: parseFloat(row.ulcer_index || 0),
        riskOfRuin: parseFloat(row.risk_of_ruin || 0),
        var_95: parseFloat(row.var_95 || 0),
        cvar_95: parseFloat(row.cvar_99 || 0), // Using cvar_99 as cvar_95
        downsideDeviation: parseFloat(row.downside_deviation || 0),
        volatility: parseFloat(row.volatility || 0),
        annualizedVolatility: parseFloat(row.annualized_volatility || 0)
      },
      trade: {
        numberOfTrades: parseFloat(row.number_of_trades || 0),
        winRate: parseFloat(row.win_rate || 0),
        lossRate: parseFloat(row.loss_rate || 0),
        averageWin: parseFloat(row.average_win || 0),
        averageLoss: parseFloat(row.average_loss || 0),
        averageTradeDuration: parseFloat(row.average_trade_duration || 0),
        largestWin: parseFloat(row.largest_win || 0),
        largestLoss: parseFloat(row.largest_loss || 0),
        consecutiveWins: parseFloat(row.consecutive_wins || 0),
        consecutiveLosses: parseFloat(row.consecutive_losses || 0),
        avgTradeReturn: parseFloat(row.avg_trade_return || 0),
        profitabilityPerTrade: parseFloat(row.profitability_per_trade || 0),
        commonSenseRatio: parseFloat(row.common_sense_ratio || 0),
        recoveryFactor: parseFloat(row.recovery_factor || 0)
      },
      profitability: {
        totalProfit: parseFloat(row.total_profit || 0),
        totalLoss: parseFloat(row.total_loss || 0),
        netProfit: parseFloat(row.net_profit || 0),
        riskReturnRatio: parseFloat(row.risk_return_ratio || 0),
        commonSenseRatio: parseFloat(row.common_sense_ratio || 0),
        conditionalDrawdownAtRisk: parseFloat(row.conditional_drawdown_at_risk || 0),
        avgProfitPerTrade: parseFloat(row.avg_profit_per_trade || 0),
        avgLossPerTrade: parseFloat(row.avg_loss_per_trade || 0),
        profitLossRatio: parseFloat(row.profit_loss_ratio || 0)
      },
      statistical: {
        skewness: parseFloat(row.skewness || 0),
        kurtosis: parseFloat(row.kurtosis || 0)
      },
      monthlyWeekly: {
        winningWeeks: parseFloat(row.winning_weeks || 0),
        losingWeeks: parseFloat(row.losing_weeks || 0),
        winningMonths: parseFloat(row.winning_months || 0),
        losingMonths: parseFloat(row.losing_months || 0),
        positiveMonthsPercent: parseFloat(row.winning_months_percent || 0),
        negativeMonthsPercent: parseFloat(row.negative_months_percent || 0)
      }
    };
    
    res.json({
      success: true,
      data: metricsData
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch metrics data',
      message: error.message
    });
  }
});

// Get ML model ledger
app.get('/api/models/:tableName/ledger', async (req, res) => {
  try {
    const tableName = req.params.tableName;
    
    // First, get the total count
    const countQuery = `
      SELECT COUNT(*) as total
      FROM ml_ledger.${tableName}
    `;
    
    const countResult = await pool.query(countQuery);
    const totalRecords = parseInt(countResult.rows[0].total);
    
    // Get all records from the ledger
    const query = `
      SELECT 
        datetime,
        action,
        buy_price,
        sell_price,
        pnl_percent,
        pnl_sum,
        balance
      FROM ml_ledger.${tableName}
      ORDER BY datetime ASC
    `;
    
    const result = await pool.query(query);
    
    const ledger = result.rows.map(row => ({
      datetime: row.datetime,
      action: row.action,
      buy_price: parseFloat(row.buy_price || 0),
      sell_price: parseFloat(row.sell_price || 0),
      pnl_percent: parseFloat(row.pnl_percent || 0),
      pnl_sum: parseFloat(row.pnl_sum || 0),
      balance: parseFloat(row.balance || 0)
    }));
    
    res.json({
      success: true,
      data: {
        ledger: ledger,
        total: totalRecords,
        totalPages: 1
      }
    });
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch ledger data',
      message: error.message
    });
  }
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({
      status: 'OK',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      database: 'Connected'
    });
  } catch (error) {
    res.status(500).json({
      status: 'ERROR',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      database: 'Disconnected',
      error: error.message
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    error: 'Something went wrong!',
    message: err.message
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found'
  });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`API available at http://localhost:${PORT}/api/strategies`);
  console.log(`Health check at http://localhost:${PORT}/api/health`);
}); 