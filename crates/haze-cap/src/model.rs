use serde::{Deserialize, Serialize};

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_empty<T>(values: &[T]) -> bool {
    values.is_empty()
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Alert {
    pub identifier: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub sender: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub sent: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub status: String,
    #[serde(rename = "message_type", default, skip_serializing_if = "is_blank")]
    pub message_type: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub scope: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub note: String,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub code: Vec<String>,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub references: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub incidents: String,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub infos: Vec<AlertInfo>,
    #[serde(rename = "raw_xml", default, skip_serializing_if = "is_blank")]
    pub raw_xml: String,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct AlertInfo {
    #[serde(default, skip_serializing_if = "is_blank")]
    pub language: String,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub category: Vec<String>,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub event: String,
    #[serde(rename = "response_type", default, skip_serializing_if = "is_empty")]
    pub response: Vec<String>,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub urgency: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub severity: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub certainty: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub audience: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub effective: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub onset: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub expires: String,
    #[serde(rename = "sender_name", default, skip_serializing_if = "is_blank")]
    pub sender_name: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub headline: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub description: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub instruction: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub web: String,
    #[serde(rename = "event_codes", default, skip_serializing_if = "is_empty")]
    pub event_codes: Vec<NameValue>,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub areas: Vec<AlertArea>,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub parameters: Vec<NameValue>,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub resources: Vec<Resource>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct AlertArea {
    #[serde(default, skip_serializing_if = "is_blank")]
    pub description: String,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub polygons: Vec<String>,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub circles: Vec<String>,
    #[serde(default, skip_serializing_if = "is_empty")]
    pub geocodes: Vec<NameValue>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NameValue {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Resource {
    #[serde(default, skip_serializing_if = "is_blank")]
    pub description: String,
    #[serde(rename = "mime_type", default, skip_serializing_if = "is_blank")]
    pub mime_type: String,
    #[serde(default, skip_serializing_if = "is_blank")]
    pub uri: String,
    #[serde(rename = "deref_uri", default, skip_serializing_if = "is_blank")]
    pub deref_uri: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AtomEntry {
    pub id: String,
    pub updated: String,
    pub links: Vec<String>,
}
