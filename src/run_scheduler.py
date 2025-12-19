import schedule
import time
import main_real_time

def job():
    import src.main_real_time
    print("Prediction done.")

schedule.every().hour.do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
