import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { Plus, Edit, Trash2, Users, Eye, EyeOff, Search, X, ChevronDown, X as XIcon, BarChart3, AlertCircle } from 'lucide-react';
import Header from './components/Header';

// Move Modal outside UserManagement
const Modal = ({ isOpen, onClose, title, onSubmit, children }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto border border-gray-200">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
};

function UserManagement() {
  const [users, setUsers] = useState([]);
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showApiSecret, setShowApiSecret] = useState(false);
  const [error, setError] = useState(null);
  const [strategySearch, setStrategySearch] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);

  // Separate form states to prevent re-renders
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [selectedStrategies, setSelectedStrategies] = useState([]);
  const [useMl, setUseMl] = useState(false);

  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const fetchUsers = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/users');
      if (response.data.success) {
        setUsers(response.data.data);
      }
    } catch (err) {
      setError('Failed to fetch users');
      console.error('Error fetching users:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchStrategies = useCallback(async () => {
    try {
      const response = await axios.get('/api/strategies');
      if (response.data.success) {
        setStrategies(response.data.data);
      }
    } catch (err) {
      console.error('Error fetching strategies:', err);
    }
  }, []);

  // Fetch users and strategies
  useEffect(() => {
    fetchUsers();
    fetchStrategies();
  }, [fetchUsers, fetchStrategies]);

  const resetFormFields = useCallback(() => {
    setName('');
    setEmail('');
    setPassword('');
    setApiKey('');
    setApiSecret('');
    setSelectedStrategies([]);
    setUseMl(false);
  }, []);

  const handleAddUser = async (e) => {
    e.preventDefault();
    try {
      setError(null);
      const response = await axios.post('/api/users', {
        name,
        email,
        password,
        api_key: apiKey,
        api_secret: apiSecret,
        strategies: selectedStrategies,
        use_ml: useMl
      });
      
      if (response.data.success) {
        setShowAddModal(false);
        resetFormFields();
        fetchUsers();
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to add user');
    }
  };

  const handleEditUser = async (e) => {
    e.preventDefault();
    try {
      setError(null);
      const response = await axios.put(`/api/users/${editingUser.id}`, {
        name,
        email,
        password,
        api_key: apiKey,
        api_secret: apiSecret,
        strategies: selectedStrategies,
        use_ml: useMl
      });
      
      if (response.data.success) {
        setShowEditModal(false);
        setEditingUser(null);
        resetFormFields();
        fetchUsers();
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to update user');
    }
  };

  const handleDeleteUser = async (userId) => {
    if (window.confirm('Are you sure you want to delete this user?')) {
      try {
        const response = await axios.delete(`/api/users/${userId}`);
        if (response.data.success) {
          fetchUsers();
          setError(null);
        }
      } catch (err) {
        setError(err.response?.data?.error || 'Failed to delete user');
      }
    }
  };

  const openEditModal = async (user) => {
    try {
      // Fetch complete user data including sensitive fields
      const response = await axios.get(`/api/users/${user.id}`);
      if (response.data.success) {
        const userData = response.data.data;
        setEditingUser(userData);
        setName(userData.name || '');
        setEmail(userData.email || '');
        setPassword(''); // Don't populate password for security
        setApiKey(userData.api_key || '');
        setApiSecret(userData.api_secret || '');
        setSelectedStrategies(parseStrategies(userData.strategies));
        setUseMl(userData.use_ml || false);
        setShowEditModal(true);
      }
    } catch (err) {
      setError('Failed to fetch user details');
      console.error('Error fetching user details:', err);
    }
  };

  const parseStrategies = (strategiesData) => {
    if (!strategiesData) return [];
    
    try {
      // If it's already an array, return it
      if (Array.isArray(strategiesData)) {
        return strategiesData;
      }
      
      // If it's a string, try to parse it as JSON
      if (typeof strategiesData === 'string') {
        return JSON.parse(strategiesData);
      }
      
      return [];
    } catch (error) {
      console.error('Error parsing strategies:', error);
      return [];
    }
  };

  // Group strategies by symbol
  const strategiesBySymbol = useMemo(() => {
    const grouped = {};
    strategies.forEach(strategy => {
      const symbol = strategy.parameters.symbol;
      if (!grouped[symbol]) {
        grouped[symbol] = [];
      }
      grouped[symbol].push(strategy);
    });
    return grouped;
  }, [strategies]);

  // Get available symbols (symbols that don't have a selected strategy)
  const availableSymbols = useMemo(() => {
    const selectedSymbols = selectedStrategies.map(strategyName => {
      const strategy = strategies.find(s => s.name === strategyName);
      return strategy ? strategy.parameters.symbol : null;
    }).filter(Boolean);
    
    return Object.keys(strategiesBySymbol).filter(symbol => 
      !selectedSymbols.includes(symbol)
    );
  }, [strategiesBySymbol, selectedStrategies, strategies]);

  const handleStrategySelect = useCallback((strategyName) => {
    const strategy = strategies.find(s => s.name === strategyName);
    if (!strategy) return;

    const symbol = strategy.parameters.symbol;
    
    // Check if this symbol already has a selected strategy
    const existingStrategyForSymbol = selectedStrategies.find(selectedStrategyName => {
      const selectedStrategy = strategies.find(s => s.name === selectedStrategyName);
      return selectedStrategy && selectedStrategy.parameters.symbol === symbol;
    });

    if (existingStrategyForSymbol) {
      // Replace the existing strategy for this symbol
      setSelectedStrategies(prev => 
        prev.filter(s => s !== existingStrategyForSymbol).concat(strategyName)
      );
    } else {
      // Add new strategy
      setSelectedStrategies(prev => [...prev, strategyName]);
    }
  }, [selectedStrategies, strategies]);

  const removeStrategy = useCallback((strategyName) => {
    setSelectedStrategies(prev => prev.filter(s => s !== strategyName));
  }, []);

  const toggleDropdown = useCallback(() => {
    setDropdownOpen(prev => !prev);
  }, []);

  // Filter strategies based on search and available symbols
  const filteredStrategies = useMemo(() => {
    let filtered = strategies;
    
    // Filter by search term
    if (strategySearch) {
      filtered = filtered.filter(strategy =>
        strategy.name.toLowerCase().includes(strategySearch.toLowerCase()) ||
        strategy.parameters.symbol.toLowerCase().includes(strategySearch.toLowerCase())
      );
    }
    
    // Only show strategies for available symbols (symbols without selected strategies)
    filtered = filtered.filter(strategy => 
      availableSymbols.includes(strategy.parameters.symbol)
    );
    
    return filtered;
  }, [strategies, strategySearch, availableSymbols]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-gray-300 border-t-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 font-medium">Loading users...</p>
          <p className="text-sm text-gray-500 mt-2">Preparing user management</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header activePage="user-management" />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">User Management</h1>
            <p className="text-gray-600 mt-1">Manage user accounts and strategy access</p>
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 shadow-sm"
          >
            <Plus className="w-4 h-4" />
            <span>Add User</span>
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center mr-3">
                <AlertCircle className="h-4 w-4 text-red-600" />
              </div>
              <div>
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Users Table */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Users</h3>
                <p className="text-sm text-gray-600 mt-1">Account management and permissions</p>
              </div>
              <div className="text-sm text-gray-600">
                {users.length} total users
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Use ML</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategies</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {users.map((user) => (
                  <tr key={user.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{user.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{user.email}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {user.use_ml ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            Enabled
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                            Disabled
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-wrap gap-1">
                        {(() => {
                          const userStrategies = parseStrategies(user.strategies);
                          return userStrategies.length > 0 ? (
                            userStrategies.map((strategy, index) => (
                              <span
                                key={index}
                                className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                              >
                                {strategy}
                              </span>
                            ))
                          ) : (
                            <span className="text-sm text-gray-500">No strategies</span>
                          );
                        })()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex justify-end space-x-2">
                        <button
                          onClick={() => openEditModal(user)}
                          className="text-blue-600 hover:text-blue-900 transition-colors p-1 rounded hover:bg-blue-50"
                          title="Edit user"
                        >
                          <Edit className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDeleteUser(user.id)}
                          className="text-red-600 hover:text-red-900 transition-colors p-1 rounded hover:bg-red-50"
                          title="Delete user"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {users.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Users className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No users found</h3>
              <p className="text-gray-500">Get started by adding your first user.</p>
            </div>
          )}
        </div>
      </main>

      {/* Add User Modal */}
      <Modal
        isOpen={showAddModal}
        onClose={() => {
          setShowAddModal(false);
          resetFormFields();
          setError(null);
        }}
        title="Add New User"
        onSubmit={handleAddUser}
      >
        <form onSubmit={handleAddUser} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Name</label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter full name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter email address"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  placeholder="Enter password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">API Key</label>
              <input
                type="text"
                required
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter API key"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">API Secret</label>
            <div className="relative">
              <input
                type={showApiSecret ? "text" : "password"}
                required
                value={apiSecret}
                onChange={(e) => setApiSecret(e.target.value)}
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter API secret"
              />
              <button
                type="button"
                onClick={() => setShowApiSecret(!showApiSecret)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
              >
                {showApiSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div>
            <label className="flex items-center space-x-2 mb-2">
              <input
                type="checkbox"
                checked={useMl}
                onChange={(e) => setUseMl(e.target.checked)}
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
              />
              <span className="text-sm font-medium text-gray-700">Use ML</span>
            </label>
            <p className="text-xs text-gray-500 mb-4">
              Enable machine learning features for this user's strategies
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Strategies (One per symbol)
            </label>
            <div className="relative" ref={dropdownRef}>
              <div className="relative">
                <div
                  onClick={toggleDropdown}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors cursor-pointer min-h-[40px] flex items-center justify-between"
                >
                  <div className="flex flex-wrap gap-1 flex-1">
                    {selectedStrategies.length > 0 ? (
                      selectedStrategies.map((strategy) => {
                        const strategyObj = strategies.find(s => s.name === strategy);
                        const symbol = strategyObj ? strategyObj.parameters.symbol : '';
                        return (
                          <span
                            key={strategy}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                          >
                            {strategy} ({symbol.toUpperCase()})
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                removeStrategy(strategy);
                              }}
                              className="ml-1 text-blue-600 hover:text-blue-800"
                            >
                              <XIcon className="w-3 h-3" />
                            </button>
                          </span>
                        );
                      })
                    ) : (
                      <span className="text-gray-500">Select strategies (one per symbol)...</span>
                    )}
                  </div>
                  <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                </div>
                
                {dropdownOpen && (
                  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                    <div className="p-2">
                      <div className="relative mb-2">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <Search className="h-4 w-4 text-gray-400" />
                        </div>
                        <input
                          type="text"
                          value={strategySearch}
                          onChange={(e) => setStrategySearch(e.target.value)}
                          className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors text-sm"
                          placeholder="Search strategies..."
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>
                      <div className="space-y-1">
                        {filteredStrategies.map((strategy) => (
                          <div
                            key={strategy.name}
                            onClick={() => handleStrategySelect(strategy.name)}
                            className={`px-3 py-2 rounded cursor-pointer transition-colors ${
                              selectedStrategies.includes(strategy.name)
                                ? 'bg-blue-100 text-blue-800'
                                : 'hover:bg-gray-100 text-gray-700'
                            }`}
                          >
                            <div className="flex justify-between items-center">
                              <span>{strategy.name}</span>
                              <span className="text-xs text-gray-500">
                                {strategy.parameters.symbol.toUpperCase()}
                              </span>
                            </div>
                          </div>
                        ))}
                        {filteredStrategies.length === 0 && (
                          <div className="px-3 py-2 text-sm text-gray-500 text-center">
                            {availableSymbols.length === 0 
                              ? 'All symbols have been selected' 
                              : 'No strategies found'
                            }
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              You can select one strategy per symbol. Available symbols: {availableSymbols.map(s => s.toUpperCase()).join(', ')}
            </p>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={() => {
                setShowAddModal(false);
                resetFormFields();
                setError(null);
              }}
              className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors"
            >
              Add User
            </button>
          </div>
        </form>
      </Modal>

      {/* Edit User Modal */}
      <Modal
        isOpen={showEditModal}
        onClose={() => {
          setShowEditModal(false);
          setEditingUser(null);
          resetFormFields();
          setError(null);
        }}
        title="Edit User"
        onSubmit={handleEditUser}
      >
        <form onSubmit={handleEditUser} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Name</label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter full name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter email address"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password (leave blank to keep current)</label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  placeholder="Enter new password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">API Key</label>
              <input
                type="text"
                required
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter API key"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">API Secret</label>
            <div className="relative">
              <input
                type={showApiSecret ? "text" : "password"}
                required
                value={apiSecret}
                onChange={(e) => setApiSecret(e.target.value)}
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter API secret"
              />
              <button
                type="button"
                onClick={() => setShowApiSecret(!showApiSecret)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
              >
                {showApiSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div>
            <label className="flex items-center space-x-2 mb-2">
              <input
                type="checkbox"
                checked={useMl}
                onChange={(e) => setUseMl(e.target.checked)}
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
              />
              <span className="text-sm font-medium text-gray-700">Use ML</span>
            </label>
            <p className="text-xs text-gray-500 mb-4">
              Enable machine learning features for this user's strategies
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Strategies (One per symbol)
            </label>
            <div className="relative" ref={dropdownRef}>
              <div className="relative">
                <div
                  onClick={toggleDropdown}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors cursor-pointer min-h-[40px] flex items-center justify-between"
                >
                  <div className="flex flex-wrap gap-1 flex-1">
                    {selectedStrategies.length > 0 ? (
                      selectedStrategies.map((strategy) => {
                        const strategyObj = strategies.find(s => s.name === strategy);
                        const symbol = strategyObj ? strategyObj.parameters.symbol : '';
                        return (
                          <span
                            key={strategy}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                          >
                            {strategy} ({symbol.toUpperCase()})
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                removeStrategy(strategy);
                              }}
                              className="ml-1 text-blue-600 hover:text-blue-800"
                            >
                              <XIcon className="w-3 h-3" />
                            </button>
                          </span>
                        );
                      })
                    ) : (
                      <span className="text-gray-500">Select strategies (one per symbol)...</span>
                    )}
                  </div>
                  <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                </div>
                
                {dropdownOpen && (
                  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                    <div className="p-2">
                      <div className="relative mb-2">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <Search className="h-4 w-4 text-gray-400" />
                        </div>
                        <input
                          type="text"
                          value={strategySearch}
                          onChange={(e) => setStrategySearch(e.target.value)}
                          className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors text-sm"
                          placeholder="Search strategies..."
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>
                      <div className="space-y-1">
                        {filteredStrategies.map((strategy) => (
                          <div
                            key={strategy.name}
                            onClick={() => handleStrategySelect(strategy.name)}
                            className={`px-3 py-2 rounded cursor-pointer transition-colors ${
                              selectedStrategies.includes(strategy.name)
                                ? 'bg-blue-100 text-blue-800'
                                : 'hover:bg-gray-100 text-gray-700'
                            }`}
                          >
                            <div className="flex justify-between items-center">
                              <span>{strategy.name}</span>
                              <span className="text-xs text-gray-500">
                                {strategy.parameters.symbol.toUpperCase()}
                              </span>
                            </div>
                          </div>
                        ))}
                        {filteredStrategies.length === 0 && (
                          <div className="px-3 py-2 text-sm text-gray-500 text-center">
                            {availableSymbols.length === 0 
                              ? 'All symbols have been selected' 
                              : 'No strategies found'
                            }
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              You can select one strategy per symbol. Available symbols: {availableSymbols.map(s => s.toUpperCase()).join(', ')}
            </p>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={() => {
                setShowEditModal(false);
                setEditingUser(null);
                resetFormFields();
                setError(null);
              }}
              className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors"
            >
              Update User
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}

export default UserManagement; 