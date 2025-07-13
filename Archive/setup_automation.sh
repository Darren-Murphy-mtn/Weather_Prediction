#!/bin/bash
# Setup script for automated daily data updates

echo "Setting up automated daily data updates for Mount Rainier Weather Prediction"

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Install required packages
echo "Installing required packages..."
pip install schedule

# Create logs directory
mkdir -p logs

# Detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - Create LaunchAgent
    echo "Setting up LaunchAgent for macOS..."
    
    LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$LAUNCH_AGENT_DIR"
    
    cat > "$LAUNCH_AGENT_DIR/com.mountrainier.weatherupdate.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mountrainier.weatherupdate</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$PROJECT_DIR/daily_data_update.py</string>
        <string>--once</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/launchagent.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/launchagent_error.log</string>
</dict>
</plist>
EOF

    # Load the LaunchAgent
    launchctl load "$LAUNCH_AGENT_DIR/com.mountrainier.weatherupdate.plist"
    echo "LaunchAgent installed and loaded"
    echo "Daily updates scheduled for 6:00 AM"
    echo "Logs will be written to logs/launchagent.log"

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - Create systemd service
    echo "Setting up systemd service for Linux..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        SYSTEMD_DIR="/etc/systemd/system"
    else
        SYSTEMD_DIR="$HOME/.config/systemd/user"
        mkdir -p "$SYSTEMD_DIR"
    fi
    
    cat > "$SYSTEMD_DIR/mountrainier-weather.service" << EOF
[Unit]
Description=Mount Rainier Weather Daily Update
After=network.target

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/python3 $PROJECT_DIR/daily_data_update.py --once
StandardOutput=append:$PROJECT_DIR/logs/systemd.log
StandardError=append:$PROJECT_DIR/logs/systemd_error.log

[Install]
WantedBy=default.target
EOF

    # Create timer
    cat > "$SYSTEMD_DIR/mountrainier-weather.timer" << EOF
[Unit]
Description=Run Mount Rainier Weather Update daily at 6 AM
Requires=mountrainier-weather.service

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Enable and start the timer
    if [[ $EUID -eq 0 ]]; then
        systemctl daemon-reload
        systemctl enable mountrainier-weather.timer
        systemctl start mountrainier-weather.timer
    else
        systemctl --user daemon-reload
        systemctl --user enable mountrainier-weather.timer
        systemctl --user start mountrainier-weather.timer
    fi
    
    echo "Systemd service installed and enabled"
    echo "Daily updates scheduled for 6:00 AM"
    echo "Logs will be written to logs/systemd.log"

else
    echo "Unsupported operating system: $OSTYPE"
    echo "Manual setup required:"
    echo "   1. Run 'python daily_data_update.py' to test"
    echo "   2. Set up cron job or task scheduler"
    echo "   3. Schedule to run daily at 6:00 AM"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "   1. Test the update: python daily_data_update.py --once"
echo "   2. Check logs in logs/ directory"
echo "   3. Monitor the first few automated runs"
echo ""
echo "Manual control:"
echo "   - Test once: python daily_data_update.py --once"
echo "   - Run scheduler: python daily_data_update.py"
echo "   - Check status: tail -f logs/daily_update.log" 