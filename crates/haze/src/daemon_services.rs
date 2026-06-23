use std::fs;
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use chrono::{Datelike, Local, Timelike, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::ServiceHostConfig;

const DEFAULT_SETTINGS_FILE: &str = "runtime/state/daemonSettings.json";
const ALERT_QUEUE_DIR: &str = "runtime/queues/alerts";

#[derive(Debug, Default, Deserialize)]
struct RootConfig {
    services: Option<ServicesConfig>,
    playout: Option<PlayoutConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct ServicesConfig {
    daemon: Option<DaemonServicesConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct DaemonServicesConfig {
    enabled: Option<bool>,
    scheduler: Option<DaemonSchedulerConfig>,
    playlist: Option<DaemonPlaylistConfig>,
    alert_queue: Option<DaemonAlertQueueConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct DaemonSchedulerConfig {
    enabled: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct DaemonPlaylistConfig {
    enabled: Option<bool>,
    interval_ms: Option<FlexibleU64>,
}

#[derive(Debug, Default, Deserialize)]
struct DaemonAlertQueueConfig {
    enabled: Option<bool>,
    interval_ms: Option<FlexibleU64>,
}

#[derive(Debug, Default, Deserialize)]
struct PlayoutConfig {
    station_id_schedule: Option<MinuteScheduleConfig>,
    date_time_schedule: Option<MinuteScheduleConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct MinuteScheduleConfig {
    enabled: Option<bool>,
    minutes: Option<MinuteList>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MinuteList {
    One(u8),
    Many(Vec<u8>),
    Text(String),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FlexibleU64 {
    Number(u64),
    Text(String),
}

impl FlexibleU64 {
    fn value(&self) -> Option<u64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::Text(value) => value.trim().parse().ok(),
        }
    }
}

pub(crate) struct DaemonServices {
    stop: Arc<AtomicBool>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl DaemonServices {
    pub(crate) fn start(
        host: &ServiceHostConfig,
        publisher: Sender<Value>,
        media_publisher: Sender<Value>,
    ) -> Result<Self> {
        let config: RootConfig = decode_config_with_overlay(&host.config_path, &host.runtime_dir)?;
        let stop = Arc::new(AtomicBool::new(false));
        let Some(daemon_cfg) = config
            .services
            .as_ref()
            .and_then(|services| services.daemon.as_ref())
        else {
            return Ok(Self {
                stop,
                threads: Vec::new(),
            });
        };
        if !daemon_cfg.enabled.unwrap_or(false) {
            return Ok(Self {
                stop,
                threads: Vec::new(),
            });
        }

        let mut threads = Vec::new();
        if daemon_cfg
            .scheduler
            .as_ref()
            .and_then(|cfg| cfg.enabled)
            .unwrap_or(false)
        {
            let schedules = SchedulePlan::from_config(config.playout.as_ref(), &host.config_path);
            let tx = publisher.clone();
            let stop_flag = Arc::clone(&stop);
            threads.push(thread::spawn(move || {
                scheduler_loop(schedules, tx, stop_flag);
            }));
            info!("daemon scheduler service enabled");
        }

        if daemon_cfg
            .alert_queue
            .as_ref()
            .and_then(|cfg| cfg.enabled)
            .unwrap_or(true)
        {
            let interval_ms = daemon_cfg
                .alert_queue
                .as_ref()
                .and_then(|cfg| cfg.interval_ms.as_ref())
                .and_then(|value| value.value())
                .unwrap_or(500)
                .clamp(100, 10_000);
            let runtime_dir = host.runtime_dir.clone();
            let tx = publisher.clone();
            let media_tx = media_publisher.clone();
            let stop_flag = Arc::clone(&stop);
            threads.push(thread::spawn(move || {
                alert_queue_loop(runtime_dir, interval_ms, tx, media_tx, stop_flag);
            }));
            info!(interval_ms, "daemon alert queue service enabled");
        }

        if daemon_cfg
            .playlist
            .as_ref()
            .and_then(|cfg| cfg.enabled)
            .unwrap_or(false)
        {
            let interval_ms = daemon_cfg
                .playlist
                .as_ref()
                .and_then(|cfg| cfg.interval_ms.as_ref())
                .and_then(|value| value.value())
                .unwrap_or(750)
                .clamp(100, 10_000);
            let tx = publisher;
            let stop_flag = Arc::clone(&stop);
            threads.push(thread::spawn(move || {
                playlist_loop(interval_ms, tx, stop_flag);
            }));
            info!(interval_ms, "daemon playlist service enabled");
        }

        Ok(Self { stop, threads })
    }

    pub(crate) fn shutdown(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        while let Some(thread) = self.threads.pop() {
            if thread.join().is_err() {
                warn!("managed service thread exited unexpectedly during shutdown");
            }
        }
    }
}

impl Drop for DaemonServices {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug, Clone)]
struct SchedulePlan {
    station_id_minutes: Vec<u8>,
    date_time_minutes: Vec<u8>,
    automations: Vec<AutomationPlan>,
}

impl SchedulePlan {
    fn from_config(playout: Option<&PlayoutConfig>, config_path: &Path) -> Self {
        let station_id = playout
            .and_then(|cfg| cfg.station_id_schedule.as_ref())
            .map(|cfg| cfg.minutes_or_default(&[0, 15, 30, 45]))
            .unwrap_or_else(|| vec![0, 15, 30, 45]);
        let date_time = playout
            .and_then(|cfg| cfg.date_time_schedule.as_ref())
            .map(|cfg| cfg.minutes_or_default(&[5, 15, 25, 35, 45, 55]))
            .unwrap_or_else(|| vec![5, 15, 25, 35, 45, 55]);
        let station_id_minutes = if playout
            .and_then(|cfg| cfg.station_id_schedule.as_ref())
            .and_then(|cfg| cfg.enabled)
            .unwrap_or(true)
        {
            station_id
        } else {
            Vec::new()
        };
        let date_time_minutes = if playout
            .and_then(|cfg| cfg.date_time_schedule.as_ref())
            .and_then(|cfg| cfg.enabled)
            .unwrap_or(true)
        {
            date_time
        } else {
            Vec::new()
        };
        Self {
            station_id_minutes,
            date_time_minutes,
            automations: load_automation_plan(config_path).unwrap_or_else(|err| {
                warn!("automation schedule unavailable: {err}");
                Vec::new()
            }),
        }
    }
}

impl MinuteScheduleConfig {
    fn minutes_or_default(&self, default: &[u8]) -> Vec<u8> {
        let mut minutes = match &self.minutes {
            Some(MinuteList::One(value)) => vec![*value],
            Some(MinuteList::Many(values)) => values.clone(),
            Some(MinuteList::Text(raw)) => raw
                .split(',')
                .filter_map(|part| part.trim().parse::<u8>().ok())
                .collect(),
            None => default.to_vec(),
        };
        minutes.retain(|minute| *minute <= 59);
        minutes.sort_unstable();
        minutes.dedup();
        if minutes.is_empty() {
            default.to_vec()
        } else {
            minutes
        }
    }
}

#[derive(Debug, Clone)]
struct AutomationPlan {
    id: String,
    name: String,
    description: String,
    schedule: AutomationSchedule,
    same: AutomationSame,
    content: AutomationContent,
    target: AutomationTarget,
}

#[derive(Debug, Clone, Default)]
struct AutomationSchedule {
    months: Vec<u32>,
    weeks: Vec<AutomationWeek>,
    days: Vec<u32>,
    weekdays: Vec<u32>,
    hours: Vec<u32>,
    minutes: Vec<u32>,
    seconds: Vec<u32>,
}

#[derive(Debug, Clone)]
struct AutomationWeek {
    week: u32,
    event_override: String,
}

#[derive(Debug, Clone)]
struct AutomationSame {
    enabled: bool,
    event: String,
    originator: String,
    locations: Vec<String>,
    duration: String,
    sender_id: String,
    tone: String,
}

#[derive(Debug, Clone)]
struct AutomationContent {
    text: String,
}

#[derive(Debug, Clone)]
struct AutomationTarget {
    feed_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AutomationsXml {
    #[serde(rename = "automation", default)]
    automations: Vec<AutomationXml>,
    #[serde(rename = "template", default)]
    templates: Vec<AutomationXml>,
}

#[derive(Debug, Deserialize)]
struct AutomationXml {
    #[serde(rename = "@id", default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    enabled: String,
    #[serde(default)]
    automated: AutomationAutomatedXml,
    #[serde(default)]
    schedule: AutomationScheduleXml,
    #[serde(default)]
    same: AutomationSameXml,
    #[serde(default)]
    content: AutomationContentXml,
    #[serde(default)]
    target: AutomationTargetXml,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationAutomatedXml {
    #[serde(default)]
    enabled: String,
    #[serde(default)]
    timing: AutomationScheduleXml,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationScheduleXml {
    #[serde(default)]
    months: String,
    #[serde(default)]
    days: String,
    #[serde(default)]
    weekdays: String,
    #[serde(default)]
    hours: String,
    #[serde(default)]
    minutes: String,
    #[serde(default)]
    seconds: String,
    #[serde(default)]
    weeks: AutomationWeeksXml,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationWeeksXml {
    #[serde(rename = "week", default)]
    weeks: Vec<AutomationWeekXml>,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationWeekXml {
    #[serde(rename = "@event_override", default)]
    event_override: String,
    #[serde(rename = "$text", default)]
    value: String,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationSameXml {
    #[serde(default)]
    enabled: String,
    #[serde(default)]
    event: String,
    #[serde(default)]
    originator: String,
    #[serde(default)]
    locations: AutomationLocationsXml,
    #[serde(default)]
    duration: AutomationDurationXml,
    #[serde(default)]
    sender_id: String,
    #[serde(default)]
    tone: String,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationLocationsXml {
    #[serde(rename = "location", default)]
    locations: Vec<AutomationLocationXml>,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationLocationXml {
    #[serde(rename = "@id", default)]
    id: String,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationDurationXml {
    #[serde(rename = "@hr", default)]
    hours: String,
    #[serde(rename = "@min", default)]
    minutes: String,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationContentXml {
    #[serde(rename = "lang", default)]
    langs: Vec<AutomationLangXml>,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationLangXml {
    #[serde(rename = "@code", default)]
    code: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationTargetXml {
    #[serde(rename = "feed", default)]
    feeds: Vec<AutomationFeedXml>,
}

#[derive(Debug, Default, Deserialize)]
struct AutomationFeedXml {
    #[serde(rename = "@id", default)]
    id: String,
}

impl AutomationPlan {
    fn matching_event(&self, now: &chrono::DateTime<Local>) -> Option<String> {
        if !matches_list(&self.schedule.months, now.month()) {
            return None;
        }
        if !matches_list(&self.schedule.days, now.day()) {
            return None;
        }
        if !matches_list(&self.schedule.weekdays, now.weekday().number_from_monday()) {
            return None;
        }
        if !matches_list(&self.schedule.hours, now.hour()) {
            return None;
        }
        if !matches_list(&self.schedule.minutes, now.minute()) {
            return None;
        }
        if !matches_list(&self.schedule.seconds, now.second()) {
            return None;
        }
        let week = week_of_month(now.day());
        if self.schedule.weeks.is_empty() {
            return Some(self.same.event.clone());
        }
        for item in &self.schedule.weeks {
            let configured_week = if item.week == 0 { 1 } else { item.week };
            if configured_week == week {
                if item.event_override.trim().is_empty() {
                    return Some(self.same.event.clone());
                }
                return Some(item.event_override.trim().to_uppercase());
            }
        }
        None
    }

    fn event_payload(&self, event: String) -> Value {
        let feed_id = if self.target.feed_ids.len() == 1 {
            self.target.feed_ids[0].clone()
        } else {
            String::new()
        };
        json!({
            "type": "cap.alert.broadcast.requested",
            "source": "automation",
            "data": {
                "alert_id": format!("automation-{}-{}", self.id, unix_now_ms()),
                "automation_id": self.id,
                "title": self.name,
                "description": self.description,
                "feed_id": feed_id,
                "feed_ids": self.target.feed_ids,
                "include_same": self.same.enabled,
                "same_originator": self.same.originator,
                "same_event": event,
                "event": event,
                "same_locations": self.same.locations,
                "same_duration": self.same.duration,
                "same_callsign": self.same.sender_id,
                "same_tone": self.same.tone,
                "alert_text": self.content.text,
                "message_type": "Alert",
                "alert_sent_at": Utc::now().to_rfc3339(),
            }
        })
    }
}

fn load_automation_plan(config_path: &Path) -> Result<Vec<AutomationPlan>> {
    let base_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let path = resolve_path(base_dir, Path::new("managed/configs/automations.xml"));
    let fallback_path = resolve_path(base_dir, Path::new("managed/configs/alertTemplates.xml"));
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            fs::read_to_string(&fallback_path).unwrap_or_default()
        }
        Err(err) => return Err(err).with_context(|| format!("failed to read {}", path.display())),
    };
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    let parsed: AutomationsXml =
        quick_xml::de::from_str(&raw).context("failed to parse automations XML")?;
    let mut plans = Vec::new();
    let mut items = parsed.automations;
    items.extend(parsed.templates);
    for (index, item) in items.into_iter().enumerate() {
        let enabled_raw = fallback_string(item.enabled.clone(), item.automated.enabled.clone());
        if !xml_bool(&enabled_raw, true) {
            continue;
        }
        let event = item.same.event.trim().to_uppercase();
        let id = fallback_string(item.id, format!("automation{}", index + 1));
        let schedule_xml = if has_schedule_xml(&item.schedule) {
            item.schedule
        } else {
            item.automated.timing
        };
        plans.push(AutomationPlan {
            id: id.clone(),
            name: fallback_string(item.name, event.clone()),
            description: item.description.trim().to_string(),
            schedule: AutomationSchedule {
                months: parse_number_list(&schedule_xml.months, 1, 12),
                days: parse_number_list(&schedule_xml.days, 1, 31),
                weekdays: parse_weekday_list(&schedule_xml.weekdays),
                hours: parse_number_list(&schedule_xml.hours, 0, 23),
                minutes: parse_number_list(&schedule_xml.minutes, 0, 59),
                seconds: parse_number_list(&schedule_xml.seconds, 0, 59),
                weeks: schedule_xml
                    .weeks
                    .weeks
                    .into_iter()
                    .filter_map(|week| {
                        let parsed = week.value.trim().parse::<u32>().ok()?;
                        if parsed <= 5 {
                            Some(AutomationWeek {
                                week: parsed,
                                event_override: week.event_override,
                            })
                        } else {
                            None
                        }
                    })
                    .collect(),
            },
            same: AutomationSame {
                enabled: xml_bool(&item.same.enabled, true),
                event,
                originator: fallback_string(item.same.originator, "WXR".to_string()).to_uppercase(),
                locations: item
                    .same
                    .locations
                    .locations
                    .into_iter()
                    .map(|location| location.id.trim().to_string())
                    .filter(|id| !id.is_empty())
                    .collect(),
                duration: automation_duration(item.same.duration),
                sender_id: item.same.sender_id.trim().to_string(),
                tone: fallback_string(item.same.tone, "WXR".to_string()).to_uppercase(),
            },
            content: AutomationContent {
                text: automation_text(item.content),
            },
            target: AutomationTarget {
                feed_ids: automation_feeds(item.target),
            },
        });
    }
    Ok(plans)
}

fn automation_duration(duration: AutomationDurationXml) -> String {
    let hours = duration.hours.trim().parse::<u32>().unwrap_or(0).min(99);
    let minutes = duration.minutes.trim().parse::<u32>().unwrap_or(15).min(59);
    format!("{hours:02}{minutes:02}")
}

fn has_schedule_xml(schedule: &AutomationScheduleXml) -> bool {
    !schedule.months.trim().is_empty()
        || !schedule.days.trim().is_empty()
        || !schedule.weekdays.trim().is_empty()
        || !schedule.hours.trim().is_empty()
        || !schedule.minutes.trim().is_empty()
        || !schedule.seconds.trim().is_empty()
        || !schedule.weeks.weeks.is_empty()
}

fn automation_text(content: AutomationContentXml) -> String {
    for lang in &content.langs {
        if lang.code.trim().eq_ignore_ascii_case("en")
            || lang.code.trim().eq_ignore_ascii_case("en-CA")
        {
            return lang.text.trim().to_string();
        }
    }
    content
        .langs
        .first()
        .map(|lang| lang.text.trim().to_string())
        .unwrap_or_default()
}

fn automation_feeds(target: AutomationTargetXml) -> Vec<String> {
    let mut feeds: Vec<String> = target
        .feeds
        .into_iter()
        .map(|feed| feed.id.trim().to_string())
        .filter(|id| !id.is_empty())
        .collect();
    if feeds.is_empty() {
        feeds.push("*".to_string());
    }
    feeds
}

fn parse_number_list(raw: &str, min: u32, max: u32) -> Vec<u32> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<u32>().ok())
        .filter(|value| *value >= min && *value <= max)
        .collect()
}

fn parse_weekday_list(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter_map(|part| match part.trim().to_ascii_lowercase().as_str() {
            "mon" | "monday" | "1" => Some(1),
            "tue" | "tues" | "tuesday" | "2" => Some(2),
            "wed" | "wednesday" | "3" => Some(3),
            "thu" | "thur" | "thurs" | "thursday" | "4" => Some(4),
            "fri" | "friday" | "5" => Some(5),
            "sat" | "saturday" | "6" => Some(6),
            "sun" | "sunday" | "7" => Some(7),
            _ => None,
        })
        .collect()
}

fn matches_list(values: &[u32], actual: u32) -> bool {
    values.is_empty() || values.contains(&actual)
}

fn week_of_month(day: u32) -> u32 {
    ((day.saturating_sub(1)) / 7) + 1
}

fn fallback_string(value: String, fallback: String) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        fallback
    } else {
        trimmed.to_string()
    }
}

fn xml_bool(raw: &str, fallback: bool) -> bool {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => fallback,
        "1" | "true" | "yes" | "on" | "enabled" => true,
        "0" | "false" | "no" | "off" | "disabled" => false,
        _ => fallback,
    }
}

fn scheduler_loop(plan: SchedulePlan, publisher: Sender<Value>, stop: Arc<AtomicBool>) {
    let mut last_minute_key = String::new();
    let mut last_second_key = String::new();
    while !stop.load(Ordering::SeqCst) {
        let now = Local::now();
        let second = now.second();
        let second_key = now.format("%Y%m%dT%H%M%S").to_string();
        if second_key != last_second_key {
            for automation in &plan.automations {
                if let Some(event) = automation.matching_event(&now) {
                    let _ = publisher.send(automation.event_payload(event));
                }
            }
            last_second_key = second_key;
        }
        if second == 0 {
            let minute = now.minute() as u8;
            let key = now.format("%Y%m%dT%H%M").to_string();
            if key != last_minute_key {
                if plan.station_id_minutes.contains(&minute) {
                    publish_scheduled_package(&publisher, "station_id");
                }
                if plan.date_time_minutes.contains(&minute) {
                    publish_scheduled_package(&publisher, "date_time");
                }
                last_minute_key = key;
            }
        }
        sleep_or_stopped(
            &stop,
            Duration::from_millis(if second == 59 { 100 } else { 250 }),
        );
    }
}

fn playlist_loop(interval_ms: u64, publisher: Sender<Value>, stop: Arc<AtomicBool>) {
    let mut first = true;
    while !stop.load(Ordering::SeqCst) {
        let message = json!({
            "type": "playlist_refill",
            "source": "daemon-playlist",
            "force": first,
            "timestamp_unix_ms": unix_now_ms(),
        });
        let _ = publisher.send(message);
        first = false;
        sleep_or_stopped(&stop, Duration::from_millis(interval_ms));
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct AlertQueueItem {
    id: String,
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    feed_ids: Vec<String>,
    #[serde(default)]
    header: String,
    #[serde(default)]
    event: String,
    #[serde(default)]
    locations: Vec<String>,
    #[serde(default)]
    duration: String,
    #[serde(default)]
    callsign: String,
    #[serde(default)]
    tone: String,
    #[serde(default)]
    audio_path: String,
    #[serde(default)]
    manifest_path: String,
    #[serde(default)]
    format: String,
    #[serde(default)]
    sample_rate: u32,
    #[serde(default)]
    channels: u16,
    #[serde(default)]
    audio_bytes: u64,
    #[serde(default)]
    source: String,
    #[serde(default)]
    outputs: Vec<AlertOutputTarget>,
    #[serde(default)]
    claimed_at: Option<String>,
    #[serde(default)]
    played_at: Option<String>,
    #[serde(default)]
    failed_at: Option<String>,
    #[serde(default)]
    last_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct AlertOutputTarget {
    #[serde(default)]
    feed_id: String,
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    address: String,
    #[serde(default)]
    format: String,
    #[serde(default)]
    acodec: String,
    #[serde(default)]
    url: String,
}

fn alert_queue_loop(
    runtime_dir: PathBuf,
    interval_ms: u64,
    publisher: Sender<Value>,
    media_publisher: Sender<Value>,
    stop: Arc<AtomicBool>,
) {
    while !stop.load(Ordering::SeqCst) {
        if let Err(err) = process_alert_queue_once_with_stop(
            &runtime_dir,
            &publisher,
            &media_publisher,
            true,
            Some(&stop),
        ) {
            warn!("alert queue processing failed: {err}");
        }
        sleep_or_stopped(&stop, Duration::from_millis(interval_ms));
    }
}

#[cfg(test)]
fn process_alert_queue_once(
    runtime_dir: &Path,
    publisher: &Sender<Value>,
    sleep_for_audio: bool,
) -> Result<usize> {
    process_alert_queue_once_with_stop(runtime_dir, publisher, publisher, sleep_for_audio, None)
}

fn process_alert_queue_once_with_stop(
    runtime_dir: &Path,
    publisher: &Sender<Value>,
    media_publisher: &Sender<Value>,
    sleep_for_audio: bool,
    stop: Option<&AtomicBool>,
) -> Result<usize> {
    let queue_dir = runtime_dir.join(ALERT_QUEUE_DIR);
    let Ok(entries) = fs::read_dir(&queue_dir) else {
        return Ok(0);
    };
    let mut manifests = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        {
            manifests.push(path);
        }
    }
    manifests.sort();

    let mut processed = 0usize;
    for manifest in manifests {
        if should_stop(stop) {
            break;
        }
        match process_alert_manifest(
            runtime_dir,
            &manifest,
            publisher,
            media_publisher,
            sleep_for_audio,
            stop,
        ) {
            Ok(true) => processed += 1,
            Ok(false) => {}
            Err(err) => {
                warn!(manifest = %manifest.display(), "failed to process alert queue item: {err}")
            }
        }
    }
    Ok(processed)
}

fn process_alert_manifest(
    runtime_dir: &Path,
    manifest: &Path,
    publisher: &Sender<Value>,
    media_publisher: &Sender<Value>,
    sleep_for_audio: bool,
    stop: Option<&AtomicBool>,
) -> Result<bool> {
    let raw = fs::read_to_string(manifest)
        .with_context(|| format!("failed to read alert manifest {}", manifest.display()))?;
    let mut item: AlertQueueItem = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse alert manifest {}", manifest.display()))?;
    if !is_pending_status(&item.status) {
        return Ok(false);
    }

    let audio_path = resolve_path(runtime_dir, Path::new(&item.audio_path));
    let audio_len = match fs::metadata(&audio_path) {
        Ok(metadata) => metadata.len(),
        Err(err) => {
            item.status = "failed".to_string();
            item.failed_at = Some(now_rfc3339());
            item.last_error = Some(format!("audio file is unavailable: {err}"));
            write_alert_manifest(manifest, &item)?;
            publish_alert_event(publisher, "alert.playout.failed", &item, Some(&audio_path));
            return Ok(true);
        }
    };
    item.status = "playing".to_string();
    item.claimed_at = Some(now_rfc3339());
    item.last_error = None;
    if item.audio_bytes == 0 {
        item.audio_bytes = audio_len;
    }
    write_alert_manifest(manifest, &item)?;
    publish_alert_event(publisher, "alert.playout.started", &item, Some(&audio_path));

    let delivered_in_realtime =
        match deliver_alert_outputs(&item, &audio_path, media_publisher, sleep_for_audio, stop) {
            Ok(value) => value,
            Err(err) => {
                item.status = "failed".to_string();
                item.failed_at = Some(now_rfc3339());
                item.last_error = Some(err.to_string());
                write_alert_manifest(manifest, &item)?;
                publish_alert_event(publisher, "alert.playout.failed", &item, Some(&audio_path));
                return Ok(true);
            }
        };

    if sleep_for_audio && !delivered_in_realtime {
        if let Some(duration) = audio_duration(&item, audio_len) {
            sleep_or_optional_stop(stop, duration);
        }
    }

    item.status = "played".to_string();
    item.played_at = Some(now_rfc3339());
    write_alert_manifest(manifest, &item)?;
    publish_alert_event(
        publisher,
        "alert.playout.completed",
        &item,
        Some(&audio_path),
    );
    Ok(true)
}

fn deliver_alert_outputs(
    item: &AlertQueueItem,
    audio_path: &Path,
    publisher: &Sender<Value>,
    sleep_for_audio: bool,
    stop: Option<&AtomicBool>,
) -> Result<bool> {
    if item.outputs.is_empty() {
        return Ok(false);
    }
    let mut delivered_in_realtime = false;
    for output in &item.outputs {
        match output.r#type.trim().to_ascii_lowercase().as_str() {
            "udp" => {
                deliver_udp_output(item, output, audio_path, sleep_for_audio, stop)?;
                delivered_in_realtime = delivered_in_realtime || sleep_for_audio;
            }
            "webrtc" | "media_bridge" => {
                deliver_media_bridge_output(
                    item,
                    output,
                    audio_path,
                    publisher,
                    sleep_for_audio,
                    stop,
                )?;
                delivered_in_realtime = delivered_in_realtime || sleep_for_audio;
            }
            "rtp" | "rtmp" | "srt" | "rtsp" | "icecast" | "stream" | "audio_device" => {
                bail!(
                    "output {} for feed {} is not supported by the alert queue worker yet",
                    output.r#type,
                    output.feed_id
                );
            }
            other => bail!(
                "output {other} for feed {} is not supported by the alert queue worker",
                output.feed_id
            ),
        }
    }
    Ok(delivered_in_realtime)
}

fn deliver_media_bridge_output(
    item: &AlertQueueItem,
    output: &AlertOutputTarget,
    audio_path: &Path,
    publisher: &Sender<Value>,
    sleep_for_audio: bool,
    stop: Option<&AtomicBool>,
) -> Result<()> {
    if !item.format.trim().eq_ignore_ascii_case("pcm_s16le") {
        bail!(
            "WebRTC output for feed {} requires pcm_s16le alert audio but manifest format is {}",
            output.feed_id,
            item.format
        );
    }
    let sample_rate = item.sample_rate.max(8_000);
    let channels = item.channels.max(1);
    let bytes_per_sample = 2usize;
    let frame_bytes = ((usize::try_from(sample_rate).unwrap_or(48_000)
        * usize::from(channels)
        * bytes_per_sample)
        / 50)
        .max(usize::from(channels) * bytes_per_sample);
    let raw = fs::read(audio_path)
        .with_context(|| format!("failed to read alert audio {}", audio_path.display()))?;
    let feed_id = if output.feed_id.trim().is_empty() {
        item.feed_ids.first().cloned().unwrap_or_default()
    } else {
        output.feed_id.clone()
    };
    if feed_id.trim().is_empty() {
        bail!("WebRTC output is missing feed_id");
    }
    for chunk in raw.chunks(frame_bytes) {
        if should_stop(stop) {
            break;
        }
        let duration_ms = pcm_chunk_duration_ms(chunk.len(), sample_rate, channels);
        let _ = publisher.send(json!({
            "type": "playout.pcm",
            "source": "daemon-alert-queue",
            "feed_id": feed_id,
            "data": {
                "feed_id": feed_id,
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_ms": duration_ms,
                "pcm": encode_base64(chunk),
            },
            "timestamp_unix_ms": unix_now_ms(),
        }));
        if sleep_for_audio {
            sleep_or_optional_stop(stop, Duration::from_millis(u64::from(duration_ms.max(1))));
        }
    }
    Ok(())
}

fn pcm_chunk_duration_ms(bytes: usize, sample_rate: u32, channels: u16) -> u32 {
    let bytes_per_second = u64::from(sample_rate.max(1)) * u64::from(channels.max(1)) * 2;
    if bytes_per_second == 0 {
        return 20;
    }
    let millis = (u64::try_from(bytes).unwrap_or(0) * 1000).div_ceil(bytes_per_second);
    u32::try_from(millis.max(1)).unwrap_or(20)
}

fn encode_base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn deliver_udp_output(
    item: &AlertQueueItem,
    output: &AlertOutputTarget,
    audio_path: &Path,
    sleep_for_audio: bool,
    stop: Option<&AtomicBool>,
) -> Result<()> {
    if !is_raw_pcm_output(output) {
        bail!(
            "udp output for feed {} requires raw PCM but is configured as format={} acodec={}",
            output.feed_id,
            output.format,
            output.acodec
        );
    }
    if output.address.trim().is_empty() {
        bail!(
            "udp output for feed {} is missing an address",
            output.feed_id
        );
    }
    let audio = fs::read(audio_path)
        .with_context(|| format!("failed to read alert audio {}", audio_path.display()))?;
    let socket = UdpSocket::bind(("0.0.0.0", 0)).context("failed to bind udp output socket")?;
    let chunk_size = udp_chunk_size(item);
    for chunk in audio.chunks(chunk_size) {
        if should_stop(stop) {
            break;
        }
        socket
            .send_to(chunk, output.address.trim())
            .with_context(|| format!("failed to send udp alert audio to {}", output.address))?;
        if sleep_for_audio {
            sleep_or_optional_stop(stop, Duration::from_millis(20));
        }
    }
    Ok(())
}

fn is_raw_pcm_output(output: &AlertOutputTarget) -> bool {
    let format = output.format.trim().to_ascii_lowercase();
    let acodec = output.acodec.trim().to_ascii_lowercase();
    matches!(format.as_str(), "" | "raw" | "pcm" | "pcm16le" | "s16le")
        && matches!(
            acodec.as_str(),
            "" | "raw" | "pcm" | "pcm_s16le" | "pcm16le" | "s16le"
        )
}

fn udp_chunk_size(item: &AlertQueueItem) -> usize {
    let sample_rate = usize::try_from(item.sample_rate).unwrap_or(48_000).max(1);
    let channels = usize::from(item.channels.max(1));
    let bytes_per_20ms = sample_rate
        .saturating_mul(channels)
        .saturating_mul(2)
        .checked_div(50)
        .unwrap_or(640);
    bytes_per_20ms.clamp(160, 1400)
}

fn write_alert_manifest(path: &Path, item: &AlertQueueItem) -> Result<()> {
    let raw = serde_json::to_vec_pretty(item).context("failed to encode alert manifest")?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, raw)
        .with_context(|| format!("failed to write alert manifest {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("failed to replace alert manifest {}", path.display()))?;
    Ok(())
}

fn publish_alert_event(
    publisher: &Sender<Value>,
    event_type: &str,
    item: &AlertQueueItem,
    audio_path: Option<&Path>,
) {
    let _ = publisher.send(json!({
        "type": event_type,
        "source": "daemon-alert-queue",
        "queue_id": &item.id,
        "feed_ids": &item.feed_ids,
        "event": &item.event,
        "header": &item.header,
        "status": &item.status,
        "audio_path": audio_path.map(|path| path.to_string_lossy().into_owned()),
        "sample_rate": item.sample_rate,
        "channels": item.channels,
        "duration_ms": audio_path
            .and_then(|path| fs::metadata(path).ok())
            .and_then(|metadata| audio_duration(item, metadata.len()))
            .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64),
        "timestamp_unix_ms": unix_now_ms(),
    }));
}

fn audio_duration(item: &AlertQueueItem, audio_len: u64) -> Option<Duration> {
    let sample_rate = u64::from(item.sample_rate);
    let channels = u64::from(item.channels.max(1));
    let bytes_per_sample = 2u64;
    let bytes_per_second = sample_rate
        .checked_mul(channels)?
        .checked_mul(bytes_per_sample)?;
    if bytes_per_second == 0 {
        return None;
    }
    let millis = audio_len.saturating_mul(1000) / bytes_per_second;
    Some(Duration::from_millis(millis.min(5 * 60 * 1000)))
}

fn is_pending_status(status: &str) -> bool {
    let normalized = status.trim().to_ascii_lowercase();
    normalized.is_empty() || normalized == "pending" || normalized == "queued"
}

fn should_stop(stop: Option<&AtomicBool>) -> bool {
    stop.is_some_and(|flag| flag.load(Ordering::SeqCst))
}

fn sleep_or_optional_stop(stop: Option<&AtomicBool>, duration: Duration) {
    if let Some(flag) = stop {
        sleep_or_stopped(flag, duration);
    } else {
        thread::sleep(duration);
    }
}

fn sleep_or_stopped(stop: &AtomicBool, duration: Duration) {
    let deadline = SystemTime::now()
        .checked_add(duration)
        .unwrap_or_else(SystemTime::now);
    while !stop.load(Ordering::SeqCst) {
        let now = SystemTime::now();
        if now >= deadline {
            break;
        }
        let remaining = deadline
            .duration_since(now)
            .unwrap_or_else(|_| Duration::from_millis(0));
        thread::sleep(remaining.min(Duration::from_millis(50)));
    }
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339()
}

fn publish_scheduled_package(publisher: &Sender<Value>, pkg_id: &str) {
    let _ = publisher.send(json!({
        "type": "scheduled_package",
        "source": "daemon-scheduler",
        "feed_id": "*",
        "pkg_id": pkg_id,
        "timestamp_unix_ms": unix_now_ms(),
    }));
}

fn decode_config_with_overlay<T>(path: &Path, runtime_dir: &Path) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read daemon config {}", path.display()))?;
    let mut value: serde_yaml::Value = serde_yaml::from_str(&raw)
        .with_context(|| format!("failed to parse daemon config {}", path.display()))?;
    let settings_file = value
        .get("daemon_settings_file")
        .and_then(serde_yaml::Value::as_str)
        .filter(|raw| !raw.trim().is_empty())
        .unwrap_or(DEFAULT_SETTINGS_FILE);
    let settings_path = resolve_path(runtime_dir, Path::new(settings_file));
    if settings_path.is_file() {
        let overlay_raw = fs::read_to_string(&settings_path).with_context(|| {
            format!(
                "failed to read daemon settings overlay {}",
                settings_path.display()
            )
        })?;
        let overlay_json: serde_json::Value =
            serde_json::from_str(&overlay_raw).with_context(|| {
                format!(
                    "failed to parse daemon settings overlay {}",
                    settings_path.display()
                )
            })?;
        merge_yaml(
            &mut value,
            serde_yaml::to_value(overlay_json)
                .context("failed to convert daemon settings overlay")?,
        );
    }
    serde_yaml::from_value(value).context("failed to decode daemon service configuration")
}

fn merge_yaml(base: &mut serde_yaml::Value, overlay: serde_yaml::Value) {
    match (base, overlay) {
        (serde_yaml::Value::Mapping(base_map), serde_yaml::Value::Mapping(overlay_map)) => {
            for (key, value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(existing) => merge_yaml(existing, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (base_slot, overlay_value) => {
            *base_slot = overlay_value;
        }
    }
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn unix_now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn alert_queue_worker_marks_pending_item_played() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = dir.path();
        let queue_dir = runtime.join(ALERT_QUEUE_DIR);
        let audio_dir = runtime.join("runtime/audio/alerts");
        fs::create_dir_all(&queue_dir).expect("queue dir");
        fs::create_dir_all(&audio_dir).expect("audio dir");
        fs::write(audio_dir.join("test.pcm16le"), [0_u8, 1, 2, 3]).expect("audio");
        fs::write(
            queue_dir.join("test.json"),
            serde_json::to_vec_pretty(&json!({
                "id": "test",
                "type": "same_alert",
                "status": "pending",
                "feed_ids": ["sk-0001"],
                "header": "ZCZC-WXR-RWT-065522+0015-1661200-XLF322  -",
                "event": "RWT",
                "audio_path": "runtime/audio/alerts/test.pcm16le",
                "format": "raw",
                "sample_rate": 48000,
                "channels": 1,
                "audio_bytes": 4
            }))
            .expect("manifest json"),
        )
        .expect("manifest");
        let (tx, rx) = mpsc::channel();

        let processed = process_alert_queue_once(runtime, &tx, false).expect("process queue");

        assert_eq!(processed, 1);
        let updated_raw = fs::read_to_string(queue_dir.join("test.json")).expect("updated");
        let updated: AlertQueueItem = serde_json::from_str(&updated_raw).expect("updated json");
        assert_eq!(updated.status, "played");
        assert!(updated.claimed_at.is_some());
        assert!(updated.played_at.is_some());
        let events: Vec<Value> = rx.try_iter().collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["type"], "alert.playout.started");
        assert_eq!(events[1]["type"], "alert.playout.completed");
    }

    #[test]
    fn alert_queue_worker_marks_missing_audio_failed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = dir.path();
        let queue_dir = runtime.join(ALERT_QUEUE_DIR);
        fs::create_dir_all(&queue_dir).expect("queue dir");
        fs::write(
            queue_dir.join("missing.json"),
            serde_json::to_vec_pretty(&json!({
                "id": "missing",
                "type": "same_alert",
                "status": "pending",
                "feed_ids": ["sk-0001"],
                "header": "ZCZC-WXR-RWT-065522+0015-1661200-XLF322  -",
                "event": "RWT",
                "audio_path": "runtime/audio/alerts/missing.pcm16le",
                "format": "raw",
                "sample_rate": 48000,
                "channels": 1
            }))
            .expect("manifest json"),
        )
        .expect("manifest");
        let (tx, rx) = mpsc::channel();

        let processed = process_alert_queue_once(runtime, &tx, false).expect("process queue");

        assert_eq!(processed, 1);
        let updated_raw = fs::read_to_string(queue_dir.join("missing.json")).expect("updated");
        let updated: AlertQueueItem = serde_json::from_str(&updated_raw).expect("updated json");
        assert_eq!(updated.status, "failed");
        assert!(updated.failed_at.is_some());
        assert!(updated.last_error.is_some());
        let events: Vec<Value> = rx.try_iter().collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "alert.playout.failed");
    }

    #[test]
    fn alert_queue_worker_sends_raw_udp_output() {
        let receiver = UdpSocket::bind(("127.0.0.1", 0)).expect("receiver");
        receiver
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("timeout");
        let address = receiver.local_addr().expect("local addr").to_string();
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = dir.path();
        let queue_dir = runtime.join(ALERT_QUEUE_DIR);
        let audio_dir = runtime.join("runtime/audio/alerts");
        fs::create_dir_all(&queue_dir).expect("queue dir");
        fs::create_dir_all(&audio_dir).expect("audio dir");
        fs::write(audio_dir.join("udp.pcm16le"), [1_u8, 2, 3, 4, 5, 6]).expect("audio");
        fs::write(
            queue_dir.join("udp.json"),
            serde_json::to_vec_pretty(&json!({
                "id": "udp",
                "type": "same_alert",
                "status": "pending",
                "feed_ids": ["sk-0001"],
                "header": "ZCZC-WXR-RWT-065522+0015-1661200-XLF322  -",
                "event": "RWT",
                "audio_path": "runtime/audio/alerts/udp.pcm16le",
                "format": "raw",
                "sample_rate": 48000,
                "channels": 1,
                "outputs": [{
                    "feed_id": "sk-0001",
                    "type": "udp",
                    "address": address,
                    "format": "raw",
                    "acodec": "pcm_s16le"
                }]
            }))
            .expect("manifest json"),
        )
        .expect("manifest");
        let (tx, _rx) = mpsc::channel();

        let processed = process_alert_queue_once(runtime, &tx, false).expect("process queue");

        assert_eq!(processed, 1);
        let mut buf = [0_u8; 32];
        let len = receiver.recv(&mut buf).expect("udp packet");
        assert_eq!(&buf[..len], &[1_u8, 2, 3, 4, 5, 6]);
        let updated_raw = fs::read_to_string(queue_dir.join("udp.json")).expect("updated");
        let updated: AlertQueueItem = serde_json::from_str(&updated_raw).expect("updated json");
        assert_eq!(updated.status, "played");
    }

    #[test]
    fn alert_queue_worker_publishes_webrtc_pcm_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = dir.path();
        let queue_dir = runtime.join(ALERT_QUEUE_DIR);
        let audio_dir = runtime.join("runtime/audio/alerts");
        fs::create_dir_all(&queue_dir).expect("queue dir");
        fs::create_dir_all(&audio_dir).expect("audio dir");
        fs::write(audio_dir.join("webrtc.pcm16le"), [1_u8, 2, 3, 4, 5, 6]).expect("audio");
        fs::write(
            queue_dir.join("webrtc.json"),
            serde_json::to_vec_pretty(&json!({
                "id": "webrtc",
                "type": "same_alert",
                "status": "pending",
                "feed_ids": ["CAP-IT-ALL"],
                "header": "ZCZC-EAS-SVR-000000+0015-1661200-CAPALL  -",
                "event": "SVR",
                "audio_path": "runtime/audio/alerts/webrtc.pcm16le",
                "format": "pcm_s16le",
                "sample_rate": 48000,
                "channels": 1,
                "audio_bytes": 6,
                "outputs": [{
                    "feed_id": "CAP-IT-ALL",
                    "type": "webrtc",
                    "format": "pcm_s16le"
                }]
            }))
            .expect("manifest json"),
        )
        .expect("manifest");
        let (tx, rx) = mpsc::channel();

        let processed = process_alert_queue_once(runtime, &tx, false).expect("process queue");

        assert_eq!(processed, 1);
        let events: Vec<Value> = rx.try_iter().collect();
        assert_eq!(events[0]["type"], "alert.playout.started");
        let pcm = events
            .iter()
            .find(|event| event["type"] == "playout.pcm")
            .expect("playout pcm event");
        assert_eq!(pcm["feed_id"], "CAP-IT-ALL");
        assert_eq!(pcm["data"]["feed_id"], "CAP-IT-ALL");
        assert_eq!(pcm["data"]["sample_rate"], 48000);
        assert_eq!(pcm["data"]["channels"], 1);
        assert_eq!(pcm["data"]["pcm"], "AQIDBAUG");
        assert_eq!(
            events.last().expect("last event")["type"],
            "alert.playout.completed"
        );
    }

    #[test]
    fn alert_queue_worker_fails_unsupported_udp_container() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = dir.path();
        let queue_dir = runtime.join(ALERT_QUEUE_DIR);
        let audio_dir = runtime.join("runtime/audio/alerts");
        fs::create_dir_all(&queue_dir).expect("queue dir");
        fs::create_dir_all(&audio_dir).expect("audio dir");
        fs::write(audio_dir.join("mpegts.pcm16le"), [0_u8, 1, 2, 3]).expect("audio");
        fs::write(
            queue_dir.join("mpegts.json"),
            serde_json::to_vec_pretty(&json!({
                "id": "mpegts",
                "type": "same_alert",
                "status": "pending",
                "feed_ids": ["sk-0001"],
                "header": "ZCZC-WXR-RWT-065522+0015-1661200-XLF322  -",
                "event": "RWT",
                "audio_path": "runtime/audio/alerts/mpegts.pcm16le",
                "format": "raw",
                "sample_rate": 48000,
                "channels": 1,
                "outputs": [{
                    "feed_id": "sk-0001",
                    "type": "udp",
                    "address": "127.0.0.1:9",
                    "format": "mpegts",
                    "acodec": "libopus"
                }]
            }))
            .expect("manifest json"),
        )
        .expect("manifest");
        let (tx, rx) = mpsc::channel();

        let processed = process_alert_queue_once(runtime, &tx, false).expect("process queue");

        assert_eq!(processed, 1);
        let updated_raw = fs::read_to_string(queue_dir.join("mpegts.json")).expect("updated");
        let updated: AlertQueueItem = serde_json::from_str(&updated_raw).expect("updated json");
        assert_eq!(updated.status, "failed");
        assert!(updated
            .last_error
            .as_deref()
            .unwrap_or_default()
            .contains("requires raw PCM"));
        let events: Vec<Value> = rx.try_iter().collect();
        assert_eq!(
            events.last().expect("event")["type"],
            "alert.playout.failed"
        );
    }
}
