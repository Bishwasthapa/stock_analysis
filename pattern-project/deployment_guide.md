# Deployment Guide: Nepal Stock Pattern Hub

Follow these steps to deploy and host this application on your private Linux server.

## 1. Prerequisites
Ensure your server has the following installed:
- **Python 3.10+**
- **Git**
- **Nginx** (for reverse proxy)
- **systemd** (standard on most Linux distros)

## 2. Server Setup

### Clone the Repository
```bash
git clone <your-repo-url>
cd pattern-project
```

### Initialize Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Data Requirements
The application relies on local CSV data in the `data/` and `results/` folders.
- Ensure you have run the analysis scripts locally and committed the results, or sync them to the server using `rsync` or `scp`.
- **Note**: The `.gitignore` may exclude large data files. You might need to manually transfer the `results/` directory.

## 4. Production Configuration

### Systemd Service (Process Management)
Create a service file to ensure the app starts automatically on boot and restarts if it crashes.

**File**: `/etc/systemd/system/pattern-hub.service`
```ini
[Unit]
Description=Nepal Stock Pattern Hub API
After=network.target

[Service]
User=<your-user>
Group=www-data
WorkingDirectory=/path/to/pattern-project
Environment="PYTHONPATH=."
ExecStart=/path/to/pattern-project/venv/bin/uvicorn code.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Commands**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pattern-hub
sudo systemctl start pattern-hub
```

### Nginx Configuration (Reverse Proxy & SSL)
Using Nginx allows you to serve the app on port 80/443 and add SSL.

**File**: `/etc/nginx/sites-available/pattern-hub`
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Commands**:
```bash
sudo ln -s /etc/nginx/sites-available/pattern-hub /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 5. Maintenance & Updates

To deploy new changes from Git:
```bash
cd /path/to/pattern-project
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart pattern-hub
```
