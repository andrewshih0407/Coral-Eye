PaleWatch


PaleWatch is a real-time monitoring system designed to track the health of coral reefs and detect early signs of bleaching. By combining a live camera feed, a machine learning model, and environmental sensors, it provides a clear, continuous view of coral conditions. Instead of waiting for visible damage, PaleWatch OS gives early warnings by analyzing both visual and environmental data, helping reef caretakers take action before stress becomes critical.

The system centers around a web dashboard that displays live video from a webcam aimed at the coral. A trained deep learning model runs in the browser to classify the coral as healthy or bleached, while an Arduino gathers environmental data such as temperature, humidity, water presence, and distance measurements. These inputs are combined into a simple reef stress score, which is displayed with color-coded indicators to show healthy, warning, or critical conditions at a glance.

The Arduino also manages physical feedback locally: LEDs indicate the stress level, a buzzer alerts during critical conditions, a servo can control vents or other mechanisms, and an LCD cycles through sensor readings. All sensor data is sent to the dashboard in real time through a Python WebSocket bridge, allowing the system to operate entirely locally without needing an internet connection. This makes it practical for remote coral nurseries or hands-on research projects.

Setting up PaleWatch OS is straightforward: upload the Arduino sketch, start the Python bridge, and open the web dashboard. From there, users can activate the camera, run live ML detection, monitor sensor readings, and capture snapshots for documentation. The combination of visual analysis and environmental monitoring creates a comprehensive and interactive way to track reef health continuously, offering both convenience and actionable insights for caretakers.

By fusing machine learning with embedded sensor data, PaleWatch OS provides a simple but powerful early warning system for coral nurseries. It’s designed to be easy to use, fully local, and focused on giving real-time feedback, helping preserve coral reefs before stress leads to bleaching or collapse.
