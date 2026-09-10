@echo off
REM Daily automated fetch of OpenPhish phishing URLs.
REM Scheduled via Windows Task Scheduler - see project README for setup steps.

call C:\Users\Akshat\anaconda3\Scripts\activate.bat phishcp
cd /d "D:\SEM - 5\ML CP (Submission)\phishing-cp"

echo. >> logs\openphish_fetch_log.txt
echo ===== Run at %date% %time% ===== >> logs\openphish_fetch_log.txt
python src\data_pipeline\fetch_openphish.py >> logs\openphish_fetch_log.txt 2>&1
