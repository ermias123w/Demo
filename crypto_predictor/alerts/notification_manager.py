import smtplib
import logging
from datetime import datetime
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
import asyncio
from dataclasses import asdict

from ..config.config import system_config, api_config
from ..models.hybrid_models import PredictionResult

# Configure logging
logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Telegram notification service"""
    
    def __init__(self):
        self.bot_token = api_config.get_api_key('TELEGRAM_BOT_TOKEN')
        self.chat_id = api_config.get_api_key('TELEGRAM_CHAT_ID')
        self.enabled = system_config.TELEGRAM_ENABLED and self.bot_token and self.chat_id
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled - missing bot token or chat ID")
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message via Telegram"""
        
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            logger.info("Telegram message sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_prediction_alert(self, prediction: PredictionResult) -> bool:
        """Send trading signal alert"""
        
        if not self.enabled:
            return False
        
        # Create message
        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        emoji = signal_emoji.get(prediction.signal, "⚪")
        
        message = f"""
<b>{emoji} {prediction.signal} Signal - {prediction.symbol}</b>

💰 <b>Entry Price:</b> ${prediction.entry_price:.2f}
🎯 <b>Take Profit:</b> ${prediction.take_profit:.2f}
🛑 <b>Stop Loss:</b> ${prediction.stop_loss:.2f}
📊 <b>Leverage:</b> {prediction.leverage:.1f}x
🎖️ <b>Confidence:</b> {prediction.confidence:.1%}

📝 <b>Rationale:</b>
{prediction.rationale}

📅 <b>Time:</b> {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(message.strip())
    
    def send_performance_alert(self, metrics: Dict) -> bool:
        """Send performance alert"""
        
        if not self.enabled:
            return False
        
        message = f"""
📊 <b>Performance Alert</b>

📈 <b>Win Rate:</b> {metrics.get('win_rate', 0):.1%}
💰 <b>Total Profit:</b> ${metrics.get('total_profit', 0):.2f}
📉 <b>Max Drawdown:</b> {metrics.get('max_drawdown', 0):.1%}
🎯 <b>Total Predictions:</b> {metrics.get('total_predictions', 0)}

📅 <b>Updated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(message.strip())
    
    def send_system_alert(self, level: str, message: str) -> bool:
        """Send system alert"""
        
        if not self.enabled:
            return False
        
        level_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨"
        }
        
        emoji = level_emoji.get(level.upper(), "📢")
        
        alert_message = f"""
{emoji} <b>System Alert - {level.upper()}</b>

{message}

📅 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(alert_message.strip())

class EmailNotifier:
    """Email notification service"""
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = api_config.get_api_key('EMAIL_USER')
        self.email_pass = api_config.get_api_key('EMAIL_PASS')
        self.enabled = system_config.EMAIL_ENABLED and self.email_user and self.email_pass
        
        if not self.enabled:
            logger.warning("Email notifications disabled - missing credentials")
    
    def send_email(self, subject: str, body: str, to_email: str = None, 
                  is_html: bool = False) -> bool:
        """Send email notification"""
        
        if not self.enabled:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = to_email or self.email_user
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_pass)
            
            text = msg.as_string()
            server.sendmail(self.email_user, to_email or self.email_user, text)
            server.quit()
            
            logger.info("Email sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_prediction_alert(self, prediction: PredictionResult) -> bool:
        """Send trading signal alert via email"""
        
        if not self.enabled:
            return False
        
        subject = f"🚀 Crypto Signal: {prediction.signal} {prediction.symbol}"
        
        body = f"""
        <html>
        <body>
        <h2>Trading Signal Alert</h2>
        
        <table border="1" style="border-collapse: collapse;">
            <tr><td><b>Symbol:</b></td><td>{prediction.symbol}</td></tr>
            <tr><td><b>Signal:</b></td><td><strong>{prediction.signal}</strong></td></tr>
            <tr><td><b>Entry Price:</b></td><td>${prediction.entry_price:.2f}</td></tr>
            <tr><td><b>Take Profit:</b></td><td>${prediction.take_profit:.2f}</td></tr>
            <tr><td><b>Stop Loss:</b></td><td>${prediction.stop_loss:.2f}</td></tr>
            <tr><td><b>Leverage:</b></td><td>{prediction.leverage:.1f}x</td></tr>
            <tr><td><b>Confidence:</b></td><td>{prediction.confidence:.1%}</td></tr>
            <tr><td><b>Time:</b></td><td>{prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        </table>
        
        <h3>Analysis:</h3>
        <p>{prediction.rationale}</p>
        
        <hr>
        <p><small>This is an automated message from the Crypto Prediction System.</small></p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body, is_html=True)
    
    def send_performance_report(self, metrics: Dict) -> bool:
        """Send performance report via email"""
        
        if not self.enabled:
            return False
        
        subject = f"📊 Crypto Predictor Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
        <html>
        <body>
        <h2>Performance Report</h2>
        
        <table border="1" style="border-collapse: collapse;">
            <tr><td><b>Total Predictions:</b></td><td>{metrics.get('total_predictions', 0)}</td></tr>
            <tr><td><b>Correct Predictions:</b></td><td>{metrics.get('correct_predictions', 0)}</td></tr>
            <tr><td><b>Win Rate:</b></td><td>{metrics.get('win_rate', 0):.1%}</td></tr>
            <tr><td><b>Total Profit:</b></td><td>${metrics.get('total_profit', 0):.2f}</td></tr>
            <tr><td><b>Max Drawdown:</b></td><td>{metrics.get('max_drawdown', 0):.1%}</td></tr>
            <tr><td><b>Sharpe Ratio:</b></td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
        </table>
        
        <h3>System Status:</h3>
        <p>All systems operational. Next update in 24 hours.</p>
        
        <hr>
        <p><small>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body, is_html=True)

class WebhookNotifier:
    """Generic webhook notification service"""
    
    def __init__(self):
        self.webhook_url = api_config.get_api_key('WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Webhook notifications disabled - no URL provided")
    
    def send_webhook(self, data: Dict) -> bool:
        """Send webhook notification"""
        
        if not self.enabled:
            return False
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'CryptoPredictorBot/1.0'
            }
            
            response = requests.post(
                self.webhook_url,
                json=data,
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            logger.info("Webhook sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    def send_prediction_alert(self, prediction: PredictionResult) -> bool:
        """Send prediction alert via webhook"""
        
        if not self.enabled:
            return False
        
        data = {
            'type': 'prediction_alert',
            'timestamp': prediction.timestamp.isoformat(),
            'symbol': prediction.symbol,
            'signal': prediction.signal,
            'confidence': prediction.confidence,
            'entry_price': prediction.entry_price,
            'stop_loss': prediction.stop_loss,
            'take_profit': prediction.take_profit,
            'leverage': prediction.leverage,
            'rationale': prediction.rationale
        }
        
        return self.send_webhook(data)

class SlackNotifier:
    """Slack notification service"""
    
    def __init__(self):
        self.webhook_url = api_config.get_api_key('SLACK_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Slack notifications disabled - no webhook URL provided")
    
    def send_message(self, message: str, channel: str = None) -> bool:
        """Send message to Slack"""
        
        if not self.enabled:
            return False
        
        try:
            data = {
                'text': message,
                'username': 'CryptoPredictorBot',
                'icon_emoji': ':robot_face:'
            }
            
            if channel:
                data['channel'] = channel
            
            response = requests.post(
                self.webhook_url,
                json=data,
                timeout=10
            )
            
            response.raise_for_status()
            logger.info("Slack message sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False
    
    def send_prediction_alert(self, prediction: PredictionResult) -> bool:
        """Send prediction alert to Slack"""
        
        if not self.enabled:
            return False
        
        signal_emoji = {"BUY": ":chart_with_upwards_trend:", "SELL": ":chart_with_downwards_trend:", "HOLD": ":pause_button:"}
        emoji = signal_emoji.get(prediction.signal, ":question:")
        
        message = f"""
{emoji} *{prediction.signal} Signal - {prediction.symbol}*

• *Entry Price:* ${prediction.entry_price:.2f}
• *Take Profit:* ${prediction.take_profit:.2f}
• *Stop Loss:* ${prediction.stop_loss:.2f}
• *Leverage:* {prediction.leverage:.1f}x
• *Confidence:* {prediction.confidence:.1%}

*Rationale:* {prediction.rationale}

_Time: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_
        """
        
        return self.send_message(message.strip())

class NotificationManager:
    """Main notification manager that coordinates all notification services"""
    
    def __init__(self):
        self.telegram = TelegramNotifier()
        self.email = EmailNotifier()
        self.webhook = WebhookNotifier()
        self.slack = SlackNotifier()
        
        self.enabled_services = []
        
        if self.telegram.enabled:
            self.enabled_services.append('telegram')
        if self.email.enabled:
            self.enabled_services.append('email')
        if self.webhook.enabled:
            self.enabled_services.append('webhook')
        if self.slack.enabled:
            self.enabled_services.append('slack')
        
        logger.info(f"Notification services enabled: {self.enabled_services}")
    
    def send_prediction_alert(self, prediction: PredictionResult) -> Dict[str, bool]:
        """Send prediction alert through all enabled services"""
        
        results = {}
        
        # Only send alerts for BUY/SELL signals
        if prediction.signal == 'HOLD':
            return results
        
        # Only send high-confidence alerts
        if prediction.confidence < 0.7:
            return results
        
        try:
            if self.telegram.enabled:
                results['telegram'] = self.telegram.send_prediction_alert(prediction)
            
            if self.email.enabled:
                results['email'] = self.email.send_prediction_alert(prediction)
            
            if self.webhook.enabled:
                results['webhook'] = self.webhook.send_prediction_alert(prediction)
            
            if self.slack.enabled:
                results['slack'] = self.slack.send_prediction_alert(prediction)
            
            success_count = sum(results.values())
            logger.info(f"Prediction alert sent via {success_count}/{len(results)} services")
            
        except Exception as e:
            logger.error(f"Error sending prediction alert: {e}")
        
        return results
    
    def send_performance_alert(self, metrics: Dict) -> Dict[str, bool]:
        """Send performance alert through enabled services"""
        
        results = {}
        
        try:
            if self.telegram.enabled:
                results['telegram'] = self.telegram.send_performance_alert(metrics)
            
            if self.email.enabled:
                results['email'] = self.email.send_performance_report(metrics)
            
            success_count = sum(results.values())
            logger.info(f"Performance alert sent via {success_count}/{len(results)} services")
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
        
        return results
    
    def send_system_alert(self, level: str, message: str) -> Dict[str, bool]:
        """Send system alert through enabled services"""
        
        results = {}
        
        # Only send critical alerts to avoid spam
        if level.upper() not in ['ERROR', 'CRITICAL']:
            return results
        
        try:
            if self.telegram.enabled:
                results['telegram'] = self.telegram.send_system_alert(level, message)
            
            if self.email.enabled:
                subject = f"🚨 System Alert - {level.upper()}"
                results['email'] = self.email.send_email(subject, message)
            
            if self.slack.enabled:
                alert_message = f":warning: *System Alert - {level.upper()}*\n\n{message}"
                results['slack'] = self.slack.send_message(alert_message)
            
            success_count = sum(results.values())
            logger.info(f"System alert sent via {success_count}/{len(results)} services")
            
        except Exception as e:
            logger.error(f"Error sending system alert: {e}")
        
        return results
    
    def send_daily_summary(self, summary: Dict) -> Dict[str, bool]:
        """Send daily summary through enabled services"""
        
        results = {}
        
        try:
            # Telegram summary
            if self.telegram.enabled:
                message = f"""
📊 <b>Daily Summary</b>

📈 <b>Predictions:</b> {summary.get('total_predictions', 0)}
🎯 <b>Accuracy:</b> {summary.get('accuracy', 0):.1%}
💰 <b>Profit:</b> ${summary.get('daily_profit', 0):.2f}
🔄 <b>Win Rate:</b> {summary.get('win_rate', 0):.1%}

📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}
                """
                results['telegram'] = self.telegram.send_message(message.strip())
            
            # Email summary
            if self.email.enabled:
                subject = f"📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
                
                body = f"""
                <html>
                <body>
                <h2>Daily Performance Summary</h2>
                
                <table border="1" style="border-collapse: collapse;">
                    <tr><td><b>Total Predictions:</b></td><td>{summary.get('total_predictions', 0)}</td></tr>
                    <tr><td><b>Accuracy:</b></td><td>{summary.get('accuracy', 0):.1%}</td></tr>
                    <tr><td><b>Daily Profit:</b></td><td>${summary.get('daily_profit', 0):.2f}</td></tr>
                    <tr><td><b>Win Rate:</b></td><td>{summary.get('win_rate', 0):.1%}</td></tr>
                </table>
                
                <h3>Top Performing Signals:</h3>
                <ul>
                    {self._format_top_signals(summary.get('top_signals', []))}
                </ul>
                
                <p><small>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
                </body>
                </html>
                """
                
                results['email'] = self.email.send_email(subject, body, is_html=True)
            
            success_count = sum(results.values())
            logger.info(f"Daily summary sent via {success_count}/{len(results)} services")
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
        
        return results
    
    def _format_top_signals(self, signals: List[Dict]) -> str:
        """Format top signals for email"""
        
        if not signals:
            return "<li>No signals today</li>"
        
        items = []
        for signal in signals[:5]:  # Top 5
            items.append(f"<li>{signal.get('symbol', 'Unknown')} - {signal.get('profit', 0):.2%} profit</li>")
        
        return '\n'.join(items)
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification services"""
        
        results = {}
        test_message = "🧪 Test notification from Crypto Predictor System"
        
        try:
            if self.telegram.enabled:
                results['telegram'] = self.telegram.send_message(test_message)
            
            if self.email.enabled:
                results['email'] = self.email.send_email("Test Email", test_message)
            
            if self.webhook.enabled:
                results['webhook'] = self.webhook.send_webhook({'message': test_message, 'type': 'test'})
            
            if self.slack.enabled:
                results['slack'] = self.slack.send_message(test_message)
            
            logger.info(f"Notification test results: {results}")
            
        except Exception as e:
            logger.error(f"Error testing notifications: {e}")
        
        return results
    
    def is_enabled(self) -> bool:
        """Check if any notification service is enabled"""
        return len(self.enabled_services) > 0
    
    def get_status(self) -> Dict:
        """Get notification services status"""
        return {
            'enabled_services': self.enabled_services,
            'telegram_enabled': self.telegram.enabled,
            'email_enabled': self.email.enabled,
            'webhook_enabled': self.webhook.enabled,
            'slack_enabled': self.slack.enabled,
            'total_services': len(self.enabled_services)
        }

# Global notification manager instance
notification_manager = NotificationManager()