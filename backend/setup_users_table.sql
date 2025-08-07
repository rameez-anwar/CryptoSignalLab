-- Create users table in the users schema
CREATE TABLE IF NOT EXISTS users.users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    api_key VARCHAR(500) NOT NULL,
    api_secret VARCHAR(500) NOT NULL,
    strategies JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users.users(email);

-- Create index on created_at for sorting
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users.users(created_at);

-- Add comments to the table
COMMENT ON TABLE users.users IS 'User management table for Crypto Signal Lab';
COMMENT ON COLUMN users.users.id IS 'Unique identifier for the user';
COMMENT ON COLUMN users.users.name IS 'Full name of the user';
COMMENT ON COLUMN users.users.email IS 'Email address (unique)';
COMMENT ON COLUMN users.users.password IS 'Hashed password (in production, use bcrypt)';
COMMENT ON COLUMN users.users.api_key IS 'Exchange API key';
COMMENT ON COLUMN users.users.api_secret IS 'Exchange API secret';
COMMENT ON COLUMN users.users.strategies IS 'JSON array of strategy names assigned to this user';
COMMENT ON COLUMN users.users.created_at IS 'Timestamp when user was created';
COMMENT ON COLUMN users.users.updated_at IS 'Timestamp when user was last updated'; 