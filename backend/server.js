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