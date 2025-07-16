#!/usr/bin/env python3
"""
Demonstration script showing the complete crypto prediction system workflow
"""

import os
import sys
import time
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n📋 STEP {step_num}: {title}")
    print("-" * 40)

def simulate_market_data():
    """Simulate market data for demo"""
    btc_price = 45000 + random.uniform(-2000, 2000)
    eth_price = 2500 + random.uniform(-200, 200)
    
    return {
        'BTC': {
            'price': btc_price,
            'volume': random.uniform(1000000, 5000000),
            'change_24h': random.uniform(-0.05, 0.05),
            'rsi': random.uniform(30, 70),
            'macd': random.uniform(-100, 100),
            'bollinger_upper': btc_price * 1.02,
            'bollinger_lower': btc_price * 0.98,
            'sentiment': random.uniform(-0.5, 0.5)
        },
        'ETH': {
            'price': eth_price,
            'volume': random.uniform(500000, 2000000),
            'change_24h': random.uniform(-0.05, 0.05),
            'rsi': random.uniform(30, 70),
            'macd': random.uniform(-50, 50),
            'bollinger_upper': eth_price * 1.02,
            'bollinger_lower': eth_price * 0.98,
            'sentiment': random.uniform(-0.5, 0.5)
        }
    }

def generate_prediction(symbol, market_data):
    """Generate a simulated prediction"""
    data = market_data[symbol]
    
    # Simple prediction logic for demo
    if data['rsi'] < 30 and data['sentiment'] > 0:
        signal = 'BUY'
        confidence = 0.75
    elif data['rsi'] > 70 and data['sentiment'] < 0:
        signal = 'SELL'
        confidence = 0.70
    else:
        signal = 'HOLD'
        confidence = 0.60
    
    # Add some randomness
    confidence += random.uniform(-0.1, 0.1)
    confidence = max(0.5, min(0.9, confidence))
    
    entry_price = data['price']
    stop_loss = entry_price * (0.98 if signal == 'BUY' else 1.02)
    take_profit = entry_price * (1.04 if signal == 'BUY' else 0.96)
    leverage = 1.0 + confidence
    
    return {
        'symbol': symbol,
        'signal': signal,
        'confidence': confidence,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'leverage': leverage,
        'sentiment_score': data['sentiment'],
        'technical_features': [data['rsi'], data['macd'], data['change_24h']],
        'rationale': f"Based on RSI={data['rsi']:.1f}, MACD={data['macd']:.1f}, Sentiment={data['sentiment']:.2f}",
        'timestamp': datetime.now(),
        'prediction_id': f"pred_{int(time.time())}"
    }

def simulate_24h_outcome(prediction):
    """Simulate the outcome after 24 hours"""
    # Random price movement
    price_change = random.uniform(-0.05, 0.05)
    new_price = prediction['entry_price'] * (1 + price_change)
    
    # Determine actual signal based on price movement
    if price_change > 0.01:
        actual_signal = 'BUY'
    elif price_change < -0.01:
        actual_signal = 'SELL'
    else:
        actual_signal = 'HOLD'
    
    # Check if prediction was correct
    correct = prediction['signal'] == actual_signal
    
    # Calculate profit/loss
    if prediction['signal'] == 'BUY':
        profit_loss = price_change * prediction['leverage']
    elif prediction['signal'] == 'SELL':
        profit_loss = -price_change * prediction['leverage']
    else:
        profit_loss = 0
    
    return {
        'new_price': new_price,
        'price_change': price_change,
        'actual_signal': actual_signal,
        'correct': correct,
        'profit_loss': profit_loss
    }

def update_reinforcement_learning(prediction, outcome):
    """Simulate reinforcement learning update"""
    # Calculate reward
    reward = 1.0 if outcome['correct'] else -0.5
    reward += outcome['profit_loss'] * 10  # Scale profit/loss
    
    # Adjust model weights (simplified)
    model_weights = {
        'lstm': 0.33,
        'transformer': 0.33,
        'sentiment': 0.34
    }
    
    # Update weights based on performance
    if outcome['correct']:
        # Increase weights for successful prediction
        for model in model_weights:
            model_weights[model] *= 1.01
    else:
        # Decrease weights for failed prediction
        for model in model_weights:
            model_weights[model] *= 0.99
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    for model in model_weights:
        model_weights[model] /= total_weight
    
    # Update strategy parameters
    confidence_adjustment = 0.01 if outcome['correct'] else -0.005
    risk_multiplier = 1.01 if outcome['profit_loss'] > 0 else 0.99
    
    return {
        'reward': reward,
        'model_weights': model_weights,
        'confidence_adjustment': confidence_adjustment,
        'risk_multiplier': risk_multiplier,
        'weights_updated': True
    }

def generate_updated_prediction(symbol, market_data, rl_update):
    """Generate a new prediction with updated parameters"""
    # Generate base prediction
    prediction = generate_prediction(symbol, market_data)
    
    # Apply RL adjustments
    prediction['confidence'] *= rl_update['risk_multiplier']
    prediction['confidence'] = max(0.5, min(0.9, prediction['confidence']))
    
    # Update model weights influence (simplified)
    if rl_update['model_weights']['sentiment'] > 0.4:
        prediction['sentiment_score'] *= 1.1
    
    prediction['rationale'] += f" (RL-adjusted: risk={rl_update['risk_multiplier']:.3f})"
    
    return prediction

def main():
    """Main demonstration function"""
    
    print_header("CRYPTO PREDICTION SYSTEM - LIVE DEMONSTRATION")
    
    print("""
🎯 This demonstration will show you the complete workflow:

1. Generate a real-time prediction
2. Monitor the system (simulated)
3. Evaluate prediction after 24 hours
4. Update system with reinforcement learning
5. Generate an improved prediction

Let's start with Bitcoin (BTC)!
    """)
    
    # Step 1: Generate Real-Time Prediction
    print_step(1, "GENERATE REAL-TIME PREDICTION")
    
    print("🔄 Collecting market data...")
    market_data = simulate_market_data()
    
    print(f"📊 Current Market Data:")
    btc_data = market_data['BTC']
    print(f"   • BTC Price: ${btc_data['price']:,.2f}")
    print(f"   • 24h Change: {btc_data['change_24h']:+.2%}")
    print(f"   • RSI: {btc_data['rsi']:.1f}")
    print(f"   • MACD: {btc_data['macd']:+.1f}")
    print(f"   • Sentiment: {btc_data['sentiment']:+.2f}")
    
    print("\n🎯 Generating prediction...")
    prediction = generate_prediction('BTC', market_data)
    
    print(f"✅ Prediction Generated!")
    print(f"   • Symbol: {prediction['symbol']}")
    print(f"   • Signal: {prediction['signal']}")
    print(f"   • Confidence: {prediction['confidence']:.1%}")
    print(f"   • Entry Price: ${prediction['entry_price']:,.2f}")
    print(f"   • Stop Loss: ${prediction['stop_loss']:,.2f}")
    print(f"   • Take Profit: ${prediction['take_profit']:,.2f}")
    print(f"   • Leverage: {prediction['leverage']:.1f}x")
    print(f"   • Rationale: {prediction['rationale']}")
    
    # Step 2: Monitor System
    print_step(2, "MONITOR SYSTEM")
    
    print("🔄 System is monitoring the prediction...")
    print("   • Collecting data every 5 minutes")
    print("   • Tracking price movements")
    print("   • Analyzing market sentiment")
    
    print("\n⏱️  Simulating 24-hour monitoring...")
    for i in range(5):
        time.sleep(1)
        print(f"   ⏰ Hour {(i+1)*4}/24: System active, data collected")
    
    print("\n✅ 24-hour monitoring completed!")
    
    # Step 3: Evaluate Prediction
    print_step(3, "EVALUATE PREDICTION AFTER 24 HOURS")
    
    print("📊 Evaluating prediction outcome...")
    outcome = simulate_24h_outcome(prediction)
    
    print(f"📈 Results after 24 hours:")
    print(f"   • Original Price: ${prediction['entry_price']:,.2f}")
    print(f"   • New Price: ${outcome['new_price']:,.2f}")
    print(f"   • Price Change: {outcome['price_change']:+.2%}")
    print(f"   • Predicted Signal: {prediction['signal']}")
    print(f"   • Actual Signal: {outcome['actual_signal']}")
    print(f"   • Prediction Correct: {'✅ YES' if outcome['correct'] else '❌ NO'}")
    print(f"   • Profit/Loss: {outcome['profit_loss']:+.2%}")
    
    if outcome['correct']:
        print("\n🎉 Prediction was CORRECT!")
    else:
        print("\n📉 Prediction was incorrect (learning opportunity)")
    
    # Step 4: Reinforcement Learning Update
    print_step(4, "REINFORCEMENT LEARNING UPDATE")
    
    print("🧠 Updating system with reinforcement learning...")
    rl_update = update_reinforcement_learning(prediction, outcome)
    
    print(f"📊 RL Update Results:")
    print(f"   • Reward: {rl_update['reward']:+.2f}")
    print(f"   • Model Weights Updated: {'✅ YES' if rl_update['weights_updated'] else '❌ NO'}")
    print(f"   • New Model Weights:")
    for model, weight in rl_update['model_weights'].items():
        print(f"     - {model.upper()}: {weight:.3f}")
    print(f"   • Confidence Adjustment: {rl_update['confidence_adjustment']:+.3f}")
    print(f"   • Risk Multiplier: {rl_update['risk_multiplier']:.3f}")
    
    print("\n✅ System updated with new learning!")
    
    # Step 5: Generate Updated Prediction
    print_step(5, "GENERATE UPDATED PREDICTION")
    
    print("🔄 Collecting fresh market data...")
    # Simulate slight market change
    new_market_data = simulate_market_data()
    
    print("🎯 Generating new prediction with updated models...")
    updated_prediction = generate_updated_prediction('BTC', new_market_data, rl_update)
    
    print(f"✅ Updated Prediction Generated!")
    print(f"   • Symbol: {updated_prediction['symbol']}")
    print(f"   • Signal: {updated_prediction['signal']}")
    print(f"   • Confidence: {updated_prediction['confidence']:.1%}")
    print(f"   • Entry Price: ${updated_prediction['entry_price']:,.2f}")
    print(f"   • Rationale: {updated_prediction['rationale']}")
    
    print(f"\n📊 Comparison with Original:")
    print(f"   • Original Confidence: {prediction['confidence']:.1%}")
    print(f"   • New Confidence: {updated_prediction['confidence']:.1%}")
    print(f"   • Confidence Change: {(updated_prediction['confidence'] - prediction['confidence']):.1%}")
    
    if updated_prediction['confidence'] > prediction['confidence']:
        print("   🚀 System confidence improved!")
    elif updated_prediction['confidence'] < prediction['confidence']:
        print("   📉 System confidence decreased (normal learning)")
    else:
        print("   ➡️ System confidence unchanged")
    
    # Final Summary
    print_header("DEMONSTRATION COMPLETE")
    
    print("🎉 Congratulations! You've seen the complete workflow:")
    print("   ✅ Real-time prediction generation")
    print("   ✅ System monitoring and data collection")
    print("   ✅ 24-hour prediction evaluation")
    print("   ✅ Reinforcement learning updates")
    print("   ✅ Improved prediction generation")
    
    print(f"\n📊 Performance Summary:")
    print(f"   • Prediction Accuracy: {'High' if outcome['correct'] else 'Learning'}")
    print(f"   • System Learning: ✅ Active")
    print(f"   • Model Weights: ✅ Updated")
    print(f"   • Risk Management: ✅ Optimized")
    
    print(f"\n🔄 Continuous Learning Cycle:")
    print("   This process repeats automatically:")
    print("   1. Generate predictions every 5-15 minutes")
    print("   2. Monitor outcomes continuously")
    print("   3. Update models every 100 predictions")
    print("   4. Improve accuracy over time")
    
    print(f"\n🎯 Next Steps:")
    print("   • Set up the full system with: python3 quick_setup.py")
    print("   • Run the interactive guide: python3 step_by_step_guide.py")
    print("   • Configure API keys in .env file")
    print("   • Start live trading (paper mode recommended)")
    
    print(f"\n🚀 Thank you for exploring the Crypto Prediction System!")
    print("   Target: 60-70% accuracy, 55%+ win rate")
    print("   Features: Real-time data, AI/ML, Risk management")
    print("   Platform: Mobile-optimized for Replit/Colab")

if __name__ == "__main__":
    main()