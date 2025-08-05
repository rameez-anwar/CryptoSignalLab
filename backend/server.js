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