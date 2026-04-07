# haze-weather-radio
A full-stack Python project that generates, manages, and broadcasts weather radio systems. It currently supports Environment Canada data feeds for weather information, and supports Pelmorex NAADS CAP-CP feeds for weather alerts. It uses Piper-TTS for text-to-speech generation, and can broadcast simultaneously on Icecast, sounddevice, and PiFmAdv locally or via SSH to a remote Raspberry Pi.

# Why
On March 16, 2026, Environment Canada shut down their Weatheradio Canada service, which provided weather radio broadcasts across the country. In response to this, I decided to create Haze Weather Radio as a replacement for the service, using publicly available data feeds and open-source tools. My goal is to provide a free and accessible weather radio service for Canadians, and to keep the spirit of Weatheradio Canada alive in a new and modernized form. As well as emphasize the importance of redundancy, accessibility, and reliability in public safety communications, and to demonstrate how technology can be used to fill gaps in public services when they arise.

# Over-The-Air Broadcasting Disclaimer
As Haze Weather Radio contains generation of valid SAME headers, it is intended for use in compliance with local regulations and broadcasting laws. Users are responsible for ensuring that their use of the software adheres to all applicable legal requirements. Broadcasting valid SAME headers runs the risk of interfering with official Emergency Alert System (EAS) equipment or weather alert radios and may be subject to legal penalties if used improperly. It is recommended to use the software for testing and educational purposes only, and to avoid broadcasting on frequencies that may interfere with official EAS broadcasts. Even if a certain someone says their transmitter only goes 10 feet.
I, the developer of Haze Weather Radio, disclaim any responsibility for misuse of the software or any legal consequences that may arise from its use.

# Features
- Generate weather radio packages with current conditions and forecasts for specified locations.
- Geophysical product used on WWV that I took from NOAA because I like it and want to use it.
- A centralized management system for multiple feeds servicing different locations.
- Modular design that allows for easy addition of new data sources, TTS engines, and output methods.
- API via FastAPI, with a web interface for monitoring, and alert origination and management.
- Support for multiple languages and localization.
- In-app support for English, French, and Spanish, with the ability to add more languages via configuration.
- Support for multiple output methods including Icecast streaming, local audio playback, and on-air broadcasting via PiFmAdv.
- Integration with official CAP feeds such as NAADS (Alert Ready/NPAS) for real-time weather alerts.
- The Weather On-Demand interface allows applications to generate custom audio packages for specific locations and conditions on demand. (e.g. An IVR system that provides weather updates for a caller's location, or a smart home device that announces weather forecasts in the morning.)

# Planned Features
- Make the web interface not look vibecoded by Claude.
- **Support for additional data sources such as NOAA, TWC, and more. (mostly finalized)**
- **Support for additional TTS engines such as PyTTSx3, eSpeak NG, Maki, and more. (pyttsx3 is now implemented, which i guess will be what the grand majority of users might want to use)**
- Be better than Weatheradio Canada.
- **Support for additional CAP feeds such as those from the US NWS (NWS-CAP ATOM, IPAWSOPEN, etc.), and international sources. (NWS-CAP has been implemented, but not thoroughly tested yet. Because I am Canadian.)**
