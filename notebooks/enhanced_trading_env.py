"""
Enhanced Trading Environment with Transaction Costs and Daily Data
"""
import numpy as np
import pandas as pd

class EnhancedTradingEnvironment:
    """
    Trading environment with:
    - Daily data support
    - Transaction costs (commission + slippage)
    - Short selling capability
    - Realistic position limits
    """
    
    def __init__(self, data, initial_capital=100000, 
                 commission_pct=0.001,  # 0.1% per trade
                 slippage_pct=0.0005,   # 0.05% slippage
                 max_position_pct=0.95,  # Max 95% capital in stock
                 enable_shorting=False):  # Enable short selling
        
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.enable_shorting = enable_shorting
        
        # State variables
        self.current_step = 0
        self.cash = initial_capital
        self.shares_held = 0
        self.portfolio_value = initial_capital
        
        # History tracking
        self.portfolio_history = [initial_capital]
        self.trade_history = []
        self.cash_history = [initial_capital]
        self.shares_history = [0]
        
        # Performance metrics
        self.total_trades = 0
        self.total_commission = 0
        self.total_slippage = 0
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares_held = 0
        self.portfolio_value = self.initial_capital
        
        self.portfolio_history = [self.initial_capital]
        self.trade_history = []
        self.cash_history = [self.initial_capital]
        self.shares_history = [0]
        
        self.total_trades = 0
        self.total_commission = 0
        self.total_slippage = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state observation"""
        row = self.data.iloc[self.current_step]
        
        # Market features (11)
        state = [
            row['close_price'],
            row['volume'],
            row['sma_5'],
            row['sma_10'],
            row['sma_20'],
            row['price_change_1d'],
            row['price_change_5d'],
            row['volatility_10d'],
            row['returns'],
            row['momentum'],
            row['volatility_ratio'],
        ]
        
        # Position features (3)
        current_price = row['close_price']
        position_value = self.shares_held * current_price
        total_value = self.cash + position_value
        
        state.extend([
            self.shares_held,
            self.cash,
            position_value / (total_value + 1e-6)  # Position ratio
        ])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Execute action and return next state
        Actions: 0=HOLD, 1=BUY, 2=SELL
        """
        current_row = self.data.iloc[self.current_step]
        current_price = current_row['close_price']
        
        # Execute action with transaction costs
        reward = 0
        trade_info = {}
        
        if action == 1:  # BUY
            reward, trade_info = self._execute_buy(current_price)
        elif action == 2:  # SELL
            reward, trade_info = self._execute_sell(current_price)
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.shares_held * current_price)
        
        # Track history
        self.portfolio_history.append(self.portfolio_value)
        self.cash_history.append(self.cash)
        self.shares_history.append(self.shares_held)
        
        if trade_info:
            self.trade_history.append({
                'step': self.current_step,
                'action': 'BUY' if action == 1 else 'SELL',
                'price': current_price,
                'shares': trade_info.get('shares', 0),
                'commission': trade_info.get('commission', 0),
                'slippage': trade_info.get('slippage', 0)
            })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        next_state = self._get_state() if not done else np.zeros(14)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares': self.shares_held,
            'total_trades': self.total_trades,
            'total_costs': self.total_commission + self.total_slippage
        }
        
        return next_state, reward, done, info
    
    def _execute_buy(self, price):
        """Execute buy order with transaction costs"""
        max_shares = int((self.cash * self.max_position_pct) / price)
        
        if max_shares <= 0:
            return 0, {}
        
        # Calculate costs
        slippage = price * self.slippage_pct
        buy_price = price + slippage
        
        trade_value = max_shares * buy_price
        commission = trade_value * self.commission_pct
        total_cost = trade_value + commission
        
        if total_cost > self.cash:
            max_shares = int(self.cash / (buy_price * (1 + self.commission_pct)))
            if max_shares <= 0:
                return 0, {}
            trade_value = max_shares * buy_price
            commission = trade_value * self.commission_pct
            total_cost = trade_value + commission
        
        # Execute trade
        self.shares_held += max_shares
        self.cash -= total_cost
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += (slippage * max_shares)
        
        # Reward: negative for costs
        reward = -(commission + slippage * max_shares) / self.initial_capital
        
        return reward, {
            'shares': max_shares,
            'commission': commission,
            'slippage': slippage * max_shares
        }
    
    def _execute_sell(self, price):
        """Execute sell order with transaction costs"""
        if self.shares_held <= 0:
            return 0, {}
        
        # Calculate costs
        slippage = price * self.slippage_pct
        sell_price = price - slippage
        
        trade_value = self.shares_held * sell_price
        commission = trade_value * self.commission_pct
        net_proceeds = trade_value - commission
        
        # Calculate profit
        cost_basis = (self.initial_capital - self.cash) / max(self.shares_held, 1)
        profit = (sell_price - cost_basis) * self.shares_held
        
        # Execute trade
        self.cash += net_proceeds
        self.shares_held = 0
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += (slippage * self.shares_held)
        
        # Reward: profit minus costs
        reward = (profit - commission - slippage * self.shares_held) / self.initial_capital
        
        return reward, {
            'shares': self.shares_held,
            'commission': commission,
            'slippage': slippage * self.shares_held
        }
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Daily Sharpe
            
            # Max drawdown
            cumulative = np.array(self.portfolio_history)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_commission + self.total_slippage,
            'final_portfolio_value': self.portfolio_value
        }

print("✓ Enhanced Trading Environment loaded")
print("  - Daily data support")
print("  - Transaction costs: 0.1% commission + 0.05% slippage")
print("  - Realistic position limits")
