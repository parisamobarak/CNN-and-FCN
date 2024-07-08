# Dockerfile
FROM python:3.12.3

#تنظیم دایرکتوری کاری برای اجرای دستورات بعدی#
WORKDIR /source

# کپی فایل requirements.txt به کانتینر
COPY requirements.txt .

# به‌روزرسانی pip
RUN pip install --upgrade pip

# نصب وابستگی‌های پایتون
RUN pip install -r requirements.txt

# کپی سایر فایل‌ها به کانتینر
COPY . .

#سرور FastAPI در پورت 8000 آماده پذیرش درخواست‌ها
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app", "--reload"]
