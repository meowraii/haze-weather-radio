package webgateway

import (
	"net/http"
	"strings"
)

func (s *Server) alertsArchiveCAPXML(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet && request.Method != http.MethodHead {
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.auth.Authenticated(request) {
		http.Error(writer, "unauthorized", http.StatusUnauthorized)
		return
	}
	id := strings.TrimSpace(request.URL.Query().Get("id"))
	feedID := strings.TrimSpace(request.URL.Query().Get("feed_id"))
	if id == "" {
		http.Error(writer, "alert id is required", http.StatusBadRequest)
		return
	}
	record, ok := findArchiveAlert(s.configPath, id, feedID)
	if !ok {
		http.NotFound(writer, request)
		return
	}
	rawXML := strings.TrimSpace(firstNonBlank(record.RawXML, record.Alert.RawXML))
	if rawXML == "" {
		http.Error(writer, "CAP XML is not available for this alert", http.StatusNotFound)
		return
	}
	writer.Header().Set("Cache-Control", "no-store")
	writer.Header().Set("Content-Type", "application/cap+xml; charset=utf-8")
	writer.Header().Set("Content-Disposition", `inline; filename="cap-`+safeID(id)+`.xml"`)
	if request.Method == http.MethodHead {
		return
	}
	_, _ = writer.Write([]byte(rawXML))
}
