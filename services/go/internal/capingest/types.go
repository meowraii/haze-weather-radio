package capingest

import "time"

// Alert is a normalized CAP alert payload produced by ingest services.
type Alert struct {
	Identifier  string      `json:"identifier"`
	Sender      string      `json:"sender,omitempty"`
	Sent        string      `json:"sent,omitempty"`
	Status      string      `json:"status,omitempty"`
	MessageType string      `json:"message_type,omitempty"`
	Scope       string      `json:"scope,omitempty"`
	Note        string      `json:"note,omitempty"`
	Code        []string    `json:"code,omitempty"`
	References  string      `json:"references,omitempty"`
	Incidents   string      `json:"incidents,omitempty"`
	Infos       []AlertInfo `json:"infos,omitempty"`
	RawXML      string      `json:"raw_xml,omitempty"`
	Warnings    []string    `json:"warnings,omitempty"`
}

// AlertInfo contains the public-safety fields Haze policy services need.
type AlertInfo struct {
	Language    string      `json:"language,omitempty"`
	Category    []string    `json:"category,omitempty"`
	Event       string      `json:"event,omitempty"`
	Response    []string    `json:"response_type,omitempty"`
	Urgency     string      `json:"urgency,omitempty"`
	Severity    string      `json:"severity,omitempty"`
	Certainty   string      `json:"certainty,omitempty"`
	Audience    string      `json:"audience,omitempty"`
	Effective   string      `json:"effective,omitempty"`
	Onset       string      `json:"onset,omitempty"`
	Expires     string      `json:"expires,omitempty"`
	SenderName  string      `json:"sender_name,omitempty"`
	Headline    string      `json:"headline,omitempty"`
	Description string      `json:"description,omitempty"`
	Instruction string      `json:"instruction,omitempty"`
	Web         string      `json:"web,omitempty"`
	EventCodes  []NameValue `json:"event_codes,omitempty"`
	Areas       []AlertArea `json:"areas,omitempty"`
	Parameters  []NameValue `json:"parameters,omitempty"`
	Resources   []Resource  `json:"resources,omitempty"`
}

// AlertArea captures CAP area metadata.
type AlertArea struct {
	Description string      `json:"description,omitempty"`
	Polygons    []string    `json:"polygons,omitempty"`
	Circles     []string    `json:"circles,omitempty"`
	Geocodes    []NameValue `json:"geocodes,omitempty"`
}

// NameValue is a generic CAP name/value pair.
type NameValue struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// Resource represents a CAP resource block.
type Resource struct {
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mime_type,omitempty"`
	URI         string `json:"uri,omitempty"`
	DerefURI    string `json:"deref_uri,omitempty"`
}

// AtomEntry is a normalized Atom feed entry.
type AtomEntry struct {
	ID      string
	Updated string
	Links   []string
}

// SourceConfig configures a CAP source.
type SourceConfig struct {
	ID           string
	URL          string
	URLs         []string
	PollInterval time.Duration
	Timeout      time.Duration
	UserAgent    string
}
