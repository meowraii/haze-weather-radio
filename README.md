# haze-weather-radio
A full-stack Python project that generates, manages, and broadcasts weather radio systems and broadcast emergency alert systems. It currently supports Environment Canada data feeds for weather information, and supports Pelmorex NAADS CAP-CP, and NWS ATOM feeds for weather alerts. It uses Piper-TTS for text-to-speech generation, and can broadcast audio streams to local network interfaces, Icecast servers, or directly to on-air broadcasting hardware while exposing a live web banner for external visual alerting. The project is designed with modularity and extensibility in mind, allowing for easy integration of additional data sources, TTS engines, and output methods in the future.

# Why
On March 16, 2026, Environment Canada shut down their Weatheradio Canada service, which provided weather radio broadcasts across the country. In response to this, I decided to create Haze Weather Radio as a replacement for the service, using publicly available data feeds and open-source tools. My goal is to provide a free and accessible weather radio service for Canadians, and to keep the spirit of Weatheradio Canada alive in a new and modernized form. As well as emphasize the importance of redundancy, accessibility, and reliability in public safety communications, and to demonstrate how technology can be used to fill gaps in public services when they arise.

# Over-The-Air Broadcasting Disclaimer
As Haze Weather Radio contains generation of valid SAME headers, it is intended for use in compliance with local regulations and broadcasting laws. Users are responsible for ensuring that their use of the software adheres to all applicable legal requirements. Broadcasting valid SAME headers runs the risk of interfering with official Emergency Alert System (EAS) equipment or weather alert radios and may be subject to legal penalties if used improperly. It is recommended to use the software for testing and educational purposes only, and to avoid broadcasting on frequencies that may interfere with official EAS broadcasts. Even if a certain someone says their transmitter only goes 10 feet.
I, the developer of Haze Weather Radio, disclaim any responsibility for misuse of the software or any legal consequences that may arise from its use.

# Features
- Weather data from ECCC, TWC, and NWS APIs.
- A centralized management system for multiple feeds servicing different locations.
- API via FastAPI, with a web interface for monitoring, and alert origination and management.
- In-app support for English, French, and Spanish, with the ability to add more languages via configuration.
- Support for multiple output methods including Icecast streaming, local audio playback, and network audio transport sinks (UDP/RTP/RTMP/SRT/RTSP).
- Integration with official CAP feeds such as NAADS (Alert Ready/NPAS) for real-time weather alerts.
- The Weather On-Demand interface allows applications to generate custom audio or text packages for specific locations and conditions on demand. (e.g. An IVR system that provides weather updates for a caller's location, or a smart home device that announces weather forecasts in the morning.)

# Planned Features
- Make the web interface not look vibecoded by Claude. **too damn lazy. if you hate vibecoded frontends, feel free to open a PR. though i did get banned from Claude if that makes you feel better.**
- Be better than Weatheradio Canada.
- **Support for additional CAP feeds such as those from the US NWS (NWS-CAP ATOM, IPAWSOPEN, etc.), and international sources. (NWS-CAP has been implemented, but not thoroughly tested yet. Because I am Canadian.)**