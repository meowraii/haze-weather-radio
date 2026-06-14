# AGENTS.md
for the Haze Weather Radio project
Written by @meowraii, June 2026.
For more information, please check out my [GitHub profile](https://github.com/meowraii).
<hr>
Haze Weather Radio is a unified, weather information and alert aggregator designed to be easy to deploy in case of situations where common infrastructure would be damaged or limited during the event of an emergency or adverse weather conditions.

It's structure is defined as an Event-Driven Architecture (EDA), a highly asynchronous execution loop where components/modules communicate by producing and consuming events via the use of an event broker, this allows for high-throughput processing and I/O, thus achieving near real-time processing and communication of public safety information. It is very much adamant that you adhere to this architecture as strictly as possible.
