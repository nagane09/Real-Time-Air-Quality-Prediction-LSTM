import smtplib
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import pandas as pd

def send_alert(predicted_pm25, threshold=100):
    if predicted_pm25 > threshold:
        msg = MIMEText(f"Warning! PM2.5 predicted to exceed safe limit: {predicted_pm25}")
        msg['Subject'] = "Air Quality Alert"
        msg['From'] = "your_email@example.com"
        msg['To'] = "user_email@example.com"
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("your_email@example.com", "YOUR_PASSWORD")
        server.send_message(msg)
        server.quit()

def plot_prediction(df, predicted_value):
    plt.plot(df['datetime'], df['pm25'], label="Historical PM2.5")
    plt.scatter(df['datetime'].iloc[-1] + pd.Timedelta(hours=1), predicted_value, color='red', label='Predicted')
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.show()
