use std::collections::HashSet;

use chrono::{DateTime, FixedOffset};
use quick_xml::de::from_str;
use serde::Deserialize;
use thiserror::Error;
use url::Url;

use crate::model::{Alert, AlertArea, AlertInfo, AtomEntry, NameValue, Resource};

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("CAP XML is not UTF-8")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("failed to parse XML: {0}")]
    Xml(#[from] quick_xml::DeError),
    #[error("invalid CAP alert {identifier}: {warnings}")]
    InvalidCap {
        identifier: String,
        warnings: String,
    },
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename = "alert")]
struct CapXml {
    #[serde(default)]
    identifier: String,
    #[serde(default)]
    sender: String,
    #[serde(default)]
    sent: String,
    #[serde(default)]
    status: String,
    #[serde(rename = "msgType", default)]
    msg_type: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    note: String,
    #[serde(rename = "code", default)]
    code: Vec<String>,
    #[serde(default)]
    references: String,
    #[serde(default)]
    incidents: String,
    #[serde(rename = "info", default)]
    infos: Vec<InfoXml>,
}

#[derive(Debug, Default, Deserialize)]
struct InfoXml {
    #[serde(default)]
    language: String,
    #[serde(rename = "category", default)]
    category: Vec<String>,
    #[serde(default)]
    event: String,
    #[serde(rename = "responseType", default)]
    response: Vec<String>,
    #[serde(default)]
    urgency: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    certainty: String,
    #[serde(default)]
    audience: String,
    #[serde(default)]
    effective: String,
    #[serde(default)]
    onset: String,
    #[serde(default)]
    expires: String,
    #[serde(rename = "senderName", default)]
    sender_name: String,
    #[serde(default)]
    headline: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    instruction: String,
    #[serde(default)]
    web: String,
    #[serde(rename = "eventCode", default)]
    event_codes: Vec<PairXml>,
    #[serde(rename = "area", default)]
    areas: Vec<AreaXml>,
    #[serde(rename = "parameter", default)]
    parameters: Vec<PairXml>,
    #[serde(rename = "resource", default)]
    resources: Vec<ResourceXml>,
}

#[derive(Debug, Default, Deserialize)]
struct AreaXml {
    #[serde(rename = "areaDesc", default)]
    description: String,
    #[serde(rename = "polygon", default)]
    polygons: Vec<String>,
    #[serde(rename = "circle", default)]
    circles: Vec<String>,
    #[serde(rename = "geocode", default)]
    geocodes: Vec<PairXml>,
}

#[derive(Debug, Default, Deserialize)]
struct PairXml {
    #[serde(rename = "valueName", default)]
    name: String,
    #[serde(default)]
    value: String,
}

#[derive(Debug, Default, Deserialize)]
struct ResourceXml {
    #[serde(rename = "resourceDesc", default)]
    description: String,
    #[serde(rename = "mimeType", default)]
    mime_type: String,
    #[serde(default)]
    uri: String,
    #[serde(rename = "derefUri", default)]
    deref_uri: String,
}

#[derive(Debug, Default, Deserialize)]
struct AtomFeedXml {
    #[serde(rename = "entry", default)]
    entries: Vec<AtomEntryXml>,
}

#[derive(Debug, Default, Deserialize)]
struct AtomEntryXml {
    #[serde(default)]
    id: String,
    #[serde(default)]
    updated: String,
    #[serde(rename = "link", default)]
    links: Vec<AtomLinkXml>,
}

#[derive(Debug, Default, Deserialize)]
struct AtomLinkXml {
    #[serde(rename = "@href", default)]
    href: String,
}

pub fn parse_cap(raw: &[u8]) -> Result<Alert, ParseError> {
    let xml = std::str::from_utf8(raw)?;
    let parsed: CapXml = from_str(xml)?;
    let mut alert = Alert {
        identifier: clean(&parsed.identifier),
        sender: clean(&parsed.sender),
        sent: clean(&parsed.sent),
        status: clean(&parsed.status),
        message_type: clean(&parsed.msg_type),
        scope: clean(&parsed.scope),
        note: clean(&parsed.note),
        code: clean_slice(parsed.code),
        references: clean(&parsed.references),
        incidents: clean(&parsed.incidents),
        infos: parsed.infos.into_iter().map(normalize_info).collect(),
        raw_xml: xml.to_string(),
        warnings: Vec::new(),
    };
    alert.warnings = validate_cap(&alert);
    if alert
        .warnings
        .iter()
        .any(|warning| warning.starts_with("fatal:"))
    {
        return Err(ParseError::InvalidCap {
            identifier: alert.identifier.clone(),
            warnings: alert.warnings.join("; "),
        });
    }
    Ok(alert)
}

pub fn parse_atom_entries(raw: &[u8]) -> Result<Vec<AtomEntry>, ParseError> {
    let xml = std::str::from_utf8(raw)?;
    let parsed: AtomFeedXml = from_str(xml)?;
    let mut entries = Vec::new();
    for entry in parsed.entries {
        let id = clean(&entry.id);
        let updated = clean(&entry.updated);
        let mut links = Vec::new();
        for link in entry.links {
            append_cap_link(&mut links, &clean(&link.href));
        }
        if id.starts_with("http") && links.is_empty() {
            append_cap_link(&mut links, &id);
        }
        if !id.is_empty() && !links.is_empty() {
            entries.push(AtomEntry { id, updated, links });
        }
    }
    Ok(entries)
}

fn normalize_info(info: InfoXml) -> AlertInfo {
    AlertInfo {
        language: clean(&info.language),
        category: clean_slice(info.category),
        event: clean(&info.event),
        response: clean_slice(info.response),
        urgency: clean(&info.urgency),
        severity: clean(&info.severity),
        certainty: clean(&info.certainty),
        audience: clean(&info.audience),
        effective: clean(&info.effective),
        onset: clean(&info.onset),
        expires: clean(&info.expires),
        sender_name: clean(&info.sender_name),
        headline: clean(&info.headline),
        description: clean(&info.description),
        instruction: clean(&info.instruction),
        web: clean(&info.web),
        event_codes: normalize_pairs(info.event_codes),
        areas: info
            .areas
            .into_iter()
            .map(|area| AlertArea {
                description: clean(&area.description),
                polygons: clean_slice(area.polygons),
                circles: clean_slice(area.circles),
                geocodes: normalize_pairs(area.geocodes),
            })
            .collect(),
        parameters: normalize_pairs(info.parameters),
        resources: info
            .resources
            .into_iter()
            .map(|resource| Resource {
                description: clean(&resource.description),
                mime_type: clean(&resource.mime_type),
                uri: clean(&resource.uri),
                deref_uri: clean(&resource.deref_uri),
            })
            .collect(),
    }
}

fn validate_cap(alert: &Alert) -> Vec<String> {
    let mut warnings = Vec::new();
    for (name, value) in [
        ("identifier", alert.identifier.as_str()),
        ("sender", alert.sender.as_str()),
        ("sent", alert.sent.as_str()),
        ("status", alert.status.as_str()),
        ("msgType", alert.message_type.as_str()),
        ("scope", alert.scope.as_str()),
    ] {
        if value.trim().is_empty() {
            warnings.push(format!("fatal: missing {name}"));
        }
    }
    if !alert.sent.is_empty() && parse_cap_time(&alert.sent).is_none() {
        warnings.push("fatal: invalid sent timestamp".to_string());
    }
    append_enum_warning(
        &mut warnings,
        "status",
        &alert.status,
        &["Actual", "Exercise", "System", "Test", "Draft"],
        true,
    );
    append_enum_warning(
        &mut warnings,
        "msgType",
        &alert.message_type,
        &["Alert", "Update", "Cancel", "Ack", "Error"],
        true,
    );
    append_enum_warning(
        &mut warnings,
        "scope",
        &alert.scope,
        &["Public", "Restricted", "Private"],
        true,
    );
    if alert.infos.is_empty()
        && !alert.message_type.eq_ignore_ascii_case("Cancel")
        && !alert.status.eq_ignore_ascii_case("System")
    {
        warnings.push("fatal: non-cancel alert has no info block".to_string());
    }
    for (index, info) in alert.infos.iter().enumerate() {
        let prefix = format!("info[{index}]");
        if info.event.is_empty() {
            warnings.push(format!("{prefix}: missing event"));
        }
        append_enum_warning(
            &mut warnings,
            &format!("{prefix}.urgency"),
            &info.urgency,
            &["Immediate", "Expected", "Future", "Past", "Unknown"],
            false,
        );
        append_enum_warning(
            &mut warnings,
            &format!("{prefix}.severity"),
            &info.severity,
            &["Extreme", "Severe", "Moderate", "Minor", "Unknown"],
            false,
        );
        append_enum_warning(
            &mut warnings,
            &format!("{prefix}.certainty"),
            &info.certainty,
            &["Observed", "Likely", "Possible", "Unlikely", "Unknown"],
            false,
        );
        for (name, value) in [
            ("effective", info.effective.as_str()),
            ("onset", info.onset.as_str()),
            ("expires", info.expires.as_str()),
        ] {
            if !value.is_empty() && parse_cap_time(value).is_none() {
                warnings.push(format!("{prefix}: invalid {name} timestamp"));
            }
        }
        for resource in &info.resources {
            if !resource.uri.is_empty() && !resource.uri.starts_with("cid:") {
                match Url::parse(&resource.uri) {
                    Ok(parsed) if !parsed.scheme().is_empty() => {}
                    _ => warnings.push(format!("{prefix}: resource URI is not absolute")),
                }
            }
        }
    }
    warnings
}

fn append_enum_warning(
    warnings: &mut Vec<String>,
    name: &str,
    value: &str,
    allowed: &[&str],
    fatal: bool,
) {
    if value.trim().is_empty() {
        return;
    }
    if allowed.iter().any(|item| value.eq_ignore_ascii_case(item)) {
        return;
    }
    if fatal {
        warnings.push(format!("fatal: invalid {name} {value}"));
    } else {
        warnings.push(format!("invalid {name} {value}"));
    }
}

fn parse_cap_time(raw: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(raw.trim()).ok()
}

fn normalize_pairs(values: Vec<PairXml>) -> Vec<NameValue> {
    values
        .into_iter()
        .filter_map(|value| {
            let name = clean(&value.name);
            let value = clean(&value.value);
            if name.is_empty() && value.is_empty() {
                None
            } else {
                Some(NameValue { name, value })
            }
        })
        .collect()
}

fn clean_slice(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| clean(&value))
        .filter(|value| !value.is_empty())
        .collect()
}

fn clean(value: &str) -> String {
    value.trim().to_string()
}

fn append_cap_link(links: &mut Vec<String>, href: &str) {
    if href.is_empty() {
        return;
    }
    let mut seen: HashSet<String> = links.iter().cloned().collect();
    if seen.insert(href.to_string()) {
        links.push(href.to_string());
    }
    if href.starts_with("http") && !href.ends_with(".cap") {
        let cap_url = format!("{}.cap", href.trim_end_matches('/'));
        if seen.insert(cap_url.clone()) {
            links.push(cap_url);
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_cap_to_go_compatible_shape() {
        let raw = br#"
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>abc</identifier>
  <sender>sender@example.test</sender>
  <sent>2026-06-22T12:38:00-06:00</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <code>profile:CAP-CP:0.4</code>
  <info>
    <language>en-CA</language>
    <category>Met</category>
    <event>Severe Thunderstorm Warning</event>
    <responseType>Shelter</responseType>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Likely</certainty>
    <senderName>Environment Canada</senderName>
    <headline>Severe thunderstorm warning</headline>
    <description>Storm text.</description>
    <instruction>Go indoors.</instruction>
    <eventCode><valueName>SAME</valueName><value>SVR</value></eventCode>
    <area>
      <areaDesc>Saskatoon</areaDesc>
      <geocode><valueName>CLC</valueName><value>065100</value></geocode>
    </area>
    <resource>
      <resourceDesc>audio</resourceDesc>
      <mimeType>audio/mpeg</mimeType>
      <uri>https://example.test/audio.mp3</uri>
    </resource>
  </info>
</alert>
"#;
        let alert = parse_cap(raw).expect("cap parsed");
        let value = serde_json::to_value(&alert).expect("json");
        assert_eq!(value["identifier"], "abc");
        assert_eq!(value["message_type"], "Alert");
        assert_eq!(value["infos"][0]["response_type"], json!(["Shelter"]));
        assert_eq!(value["infos"][0]["event_codes"][0]["value"], "SVR");
        assert_eq!(
            value["infos"][0]["areas"][0]["geocodes"][0]["value"],
            "065100"
        );
    }

    #[test]
    fn parses_atom_links_and_cap_fallbacks() {
        let raw = br#"
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://alerts.example/item/1</id>
    <updated>2026-06-22T12:39:00Z</updated>
    <link href="https://alerts.example/item/1"/>
  </entry>
</feed>
"#;
        let entries = parse_atom_entries(raw).expect("atom parsed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "https://alerts.example/item/1");
        assert_eq!(entries[0].links[0], "https://alerts.example/item/1");
        assert_eq!(entries[0].links[1], "https://alerts.example/item/1.cap");
    }

    #[test]
    fn accepts_naads_heartbeat_without_info() {
        let raw = br#"
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>heartbeat-1</identifier>
  <sender>NAADS-Heartbeat</sender>
  <sent>2026-06-22T12:38:00-06:00</sent>
  <status>System</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <references>sender,abc,2026-06-22T12:37:00-06:00</references>
</alert>
"#;
        let alert = parse_cap(raw).expect("heartbeat parsed");
        assert_eq!(alert.status, "System");
        assert!(alert.infos.is_empty());
    }
}
