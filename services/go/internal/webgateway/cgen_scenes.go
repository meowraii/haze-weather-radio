package webgateway

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

const (
	cgenScenesRelativeDir = "managed/cgen/scenes"
	cgenSceneMaxXMLBytes  = 512 * 1024
	cgenSceneMaxCount     = 1024
	cgenSceneMaxNodeCount = 4096
	cgenSceneMaxNodeDepth = 32
	cgenSceneMaxXMLDepth  = 128
	cgenSceneSchema       = "1"
)

var cgenScenesMu sync.RWMutex

type cgenProtectedScene struct {
	ID       string
	Name     string
	Filename string
	Locked   bool
}

var cgenProtectedScenes = []cgenProtectedScene{
	{ID: "Program_Passthrough", Name: "Program_Passthrough", Filename: "program_passthrough.xml", Locked: true},
	{ID: "Standard_Crawl", Name: "Standard_Crawl", Filename: "crawl.xml"},
	{ID: "Fullscreen_Takeover", Name: "Fullscreen_Takeover", Filename: "fullscreen.xml"},
	{ID: "Standby", Name: "Standby", Filename: "standby.xml"},
}

type cgenSceneXMLMetadata struct {
	XMLName       xml.Name
	SchemaVersion string `xml:"schema_version,attr"`
	ID            string `xml:"id,attr"`
	Name          string `xml:"name,attr"`
}

type cgenSceneRecord struct {
	Metadata  cgenSceneXMLMetadata
	Filename  string
	Revision  string
	XML       []byte
	Protected bool
	Locked    bool
}

func listCgenScenesPayload(configPath string) (map[string]any, error) {
	cgenScenesMu.RLock()
	defer cgenScenesMu.RUnlock()

	root, exists, err := openCgenScenesRoot(configPath, false)
	if err != nil {
		return nil, err
	}
	if !exists {
		return map[string]any{
			"scenes":   []map[string]any{},
			"revision": cgenSceneCollectionRevision(nil),
		}, nil
	}
	defer func() { _ = root.Close() }()

	records, err := readAllCgenSceneRecords(root)
	if err != nil {
		return nil, err
	}
	return cgenSceneListPayload(records), nil
}

func getCgenScenePayload(configPath string, payload map[string]any) (map[string]any, error) {
	cgenScenesMu.RLock()
	defer cgenScenesMu.RUnlock()

	root, exists, err := openCgenScenesRoot(configPath, false)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.New("scene was not found")
	}
	defer func() { _ = root.Close() }()

	records, err := readAllCgenSceneRecords(root)
	if err != nil {
		return nil, err
	}
	record, ok, err := selectCgenSceneRecord(records, payload)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, errors.New("scene was not found")
	}
	scene := cgenSceneRecordPayload(record)
	scene["xml"] = string(record.XML)
	return map[string]any{"scene": scene}, nil
}

func saveCgenScenePayload(configPath string, payload map[string]any) (map[string]any, error) {
	cgenScenesMu.Lock()
	defer cgenScenesMu.Unlock()

	expectedRevision, err := requiredCgenSceneRevision(payload)
	if err != nil {
		return nil, err
	}
	rawXML, err := cgenSceneXMLPayload(payload)
	if err != nil {
		return nil, err
	}
	metadata, err := parseCgenSceneMetadata(rawXML)
	if err != nil {
		return nil, err
	}
	if requestedID := strings.TrimSpace(firstNonBlank(stringValue(payload, "scene_id"), stringValue(payload, "id"))); requestedID != "" {
		if !validCgenSceneID(requestedID) || requestedID != metadata.ID {
			return nil, errors.New("scene id does not match the XML document")
		}
	}

	root, _, err := openCgenScenesRoot(configPath, true)
	if err != nil {
		return nil, err
	}
	defer func() { _ = root.Close() }()

	records, err := readAllCgenSceneRecords(root)
	if err != nil {
		return nil, err
	}
	originalID := strings.TrimSpace(stringValue(payload, "original_id"))
	if originalID != "" && !validCgenSceneID(originalID) {
		return nil, errors.New("original scene id is invalid")
	}

	var source cgenSceneRecord
	sourceFound := false
	if originalID != "" {
		source, sourceFound = findCgenSceneRecordByID(records, originalID)
		if !sourceFound {
			return nil, errors.New("scene was not found")
		}
	} else {
		source, sourceFound = findCgenSceneRecordByID(records, metadata.ID)
	}
	if sourceFound && source.Locked {
		return nil, errors.New("Program_Passthrough is locked")
	}
	if sourceFound && source.Protected {
		if source.Metadata.ID != metadata.ID || source.Metadata.Name != metadata.Name {
			return nil, errors.New("protected scenes cannot be renamed")
		}
	}
	if protected, ok := protectedCgenSceneByID(metadata.ID); ok {
		if protected.Locked {
			return nil, errors.New("Program_Passthrough is locked")
		}
		if metadata.Name != protected.Name {
			return nil, errors.New("protected scenes cannot be renamed")
		}
	}

	if sourceFound {
		if expectedRevision != source.Revision {
			return nil, errors.New("scene revision conflict")
		}
	} else if expectedRevision != "" {
		return nil, errors.New("scene revision conflict")
	}

	targetFilename, err := cgenSceneTargetFilename(metadata.ID, stringValue(payload, "filename"), source, sourceFound)
	if err != nil {
		return nil, err
	}
	if sourceFound && source.Protected && source.Filename != targetFilename {
		return nil, errors.New("protected scenes cannot be renamed")
	}
	if other, ok := findCgenSceneRecordByFilename(records, targetFilename); ok && (!sourceFound || other.Filename != source.Filename) {
		return nil, errors.New("scene filename is already in use")
	}
	if other, ok := findCgenSceneRecordByID(records, metadata.ID); ok && (!sourceFound || other.Filename != source.Filename) {
		return nil, errors.New("scene id is already in use")
	}

	if err := writeCgenSceneAtomic(root, targetFilename, rawXML); err != nil {
		return nil, err
	}
	if sourceFound && !strings.EqualFold(source.Filename, targetFilename) {
		if err := root.Remove(source.Filename); err != nil {
			return nil, errors.New("scene storage could not complete the rename")
		}
	}
	record := cgenSceneRecordFromRaw(targetFilename, metadata, rawXML)
	result := cgenSceneRecordPayload(record)
	result["changed_scene_id"] = metadata.ID
	result["saved"] = true
	return result, nil
}

func deleteCgenScenePayload(configPath string, payload map[string]any) (map[string]any, error) {
	cgenScenesMu.Lock()
	defer cgenScenesMu.Unlock()

	expectedRevision, err := requiredCgenSceneRevision(payload)
	if err != nil {
		return nil, err
	}
	root, exists, err := openCgenScenesRoot(configPath, false)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.New("scene was not found")
	}
	defer func() { _ = root.Close() }()

	records, err := readAllCgenSceneRecords(root)
	if err != nil {
		return nil, err
	}
	record, ok, err := selectCgenSceneRecord(records, payload)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, errors.New("scene was not found")
	}
	if record.Protected {
		return nil, errors.New("protected scenes cannot be deleted")
	}
	if expectedRevision != record.Revision {
		return nil, errors.New("scene revision conflict")
	}
	if err := root.Remove(record.Filename); err != nil {
		return nil, errors.New("scene could not be deleted")
	}

	remaining := make([]cgenSceneRecord, 0, len(records)-1)
	for _, item := range records {
		if item.Filename != record.Filename {
			remaining = append(remaining, item)
		}
	}
	return map[string]any{
		"changed_scene_id":  record.Metadata.ID,
		"deleted":           true,
		"previous_revision": record.Revision,
		"revision":          cgenSceneCollectionRevision(remaining),
	}, nil
}

func (s *wsSession) publishCgenScenesUpdated(payload map[string]any) error {
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil
	}
	publisher := events.NewHostBridgePublisher(bridgeAddr)
	defer publisher.Close()
	return publisher.Publish(events.Event{
		Type:    "cgen.scenes.updated",
		Source:  "haze-web",
		Subject: strings.TrimSpace(stringValue(payload, "changed_scene_id")),
		Data:    payload,
	})
}

func openCgenScenesRoot(configPath string, create bool) (*os.Root, bool, error) {
	base := filepath.Dir(filepath.Clean(configPath))
	baseAbs, err := filepath.Abs(base)
	if err != nil {
		return nil, false, errors.New("scene storage is unavailable")
	}
	projectRoot, err := os.OpenRoot(baseAbs)
	if err != nil {
		return nil, false, errors.New("scene storage is unavailable")
	}
	defer func() { _ = projectRoot.Close() }()

	relativeDir := filepath.FromSlash(cgenScenesRelativeDir)
	if create {
		if err := projectRoot.MkdirAll(relativeDir, 0o700); err != nil {
			return nil, false, errors.New("scene storage is unavailable")
		}
	}
	root, err := projectRoot.OpenRoot(relativeDir)
	if err != nil {
		if !create && os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, errors.New("scene storage is unavailable")
	}
	return root, true, nil
}

func readAllCgenSceneRecords(root *os.Root) ([]cgenSceneRecord, error) {
	dir, err := root.Open(".")
	if err != nil {
		return nil, errors.New("scene storage is unavailable")
	}
	entries, err := dir.ReadDir(cgenSceneMaxCount + 1)
	closeErr := dir.Close()
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, errors.New("scene storage is unavailable")
	}
	if closeErr != nil {
		return nil, errors.New("scene storage is unavailable")
	}
	if len(entries) > cgenSceneMaxCount {
		return nil, errors.New("scene storage contains too many entries")
	}

	records := make([]cgenSceneRecord, 0, len(entries))
	seenIDs := make(map[string]struct{}, len(entries))
	seenFilenames := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		filename := entry.Name()
		if !strings.HasSuffix(strings.ToLower(filename), ".xml") {
			continue
		}
		if !validCgenSceneFilename(filename) {
			continue
		}
		portableFilename := strings.ToLower(filename)
		if _, exists := seenFilenames[portableFilename]; exists {
			return nil, errors.New("scene storage contains colliding filenames")
		}
		seenFilenames[portableFilename] = struct{}{}
		info, err := root.Lstat(filename)
		if err != nil {
			return nil, errors.New("scene storage changed while it was being read")
		}
		if !info.Mode().IsRegular() {
			return nil, errors.New("scene storage contains an unsafe entry")
		}
		raw, err := readCgenSceneFile(root, filename)
		if err != nil {
			return nil, err
		}
		metadata, err := parseCgenSceneMetadata(raw)
		if err != nil {
			return nil, errors.New("scene storage contains an invalid XML document")
		}
		record := cgenSceneRecordFromRaw(filename, metadata, raw)
		if err := validateCgenSceneRecordIdentity(record); err != nil {
			return nil, err
		}
		if _, exists := seenIDs[metadata.ID]; exists {
			return nil, errors.New("scene storage contains duplicate scene ids")
		}
		seenIDs[metadata.ID] = struct{}{}
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool {
		return records[i].Metadata.ID < records[j].Metadata.ID
	})
	return records, nil
}

func readCgenSceneFile(root *os.Root, filename string) ([]byte, error) {
	file, err := root.Open(filename)
	if err != nil {
		return nil, errors.New("scene could not be read")
	}
	raw, readErr := io.ReadAll(io.LimitReader(file, cgenSceneMaxXMLBytes+1))
	closeErr := file.Close()
	if readErr != nil || closeErr != nil {
		return nil, errors.New("scene could not be read")
	}
	if len(raw) > cgenSceneMaxXMLBytes {
		return nil, errors.New("scene XML exceeds the size limit")
	}
	return raw, nil
}

func parseCgenSceneMetadata(raw []byte) (cgenSceneXMLMetadata, error) {
	if len(raw) == 0 {
		return cgenSceneXMLMetadata{}, errors.New("scene XML is required")
	}
	if len(raw) > cgenSceneMaxXMLBytes {
		return cgenSceneXMLMetadata{}, errors.New("scene XML exceeds the size limit")
	}
	decoder := xml.NewDecoder(strings.NewReader(string(raw)))
	decoder.Strict = true
	var metadata cgenSceneXMLMetadata
	rootSeen := false
	depth := 0
	nodeDepth := 0
	nodeCount := 0
	rootNodeCount := 0
	nodeIDs := make(map[string]struct{})
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return cgenSceneXMLMetadata{}, errors.New("scene XML is invalid")
		}
		switch typed := token.(type) {
		case xml.Directive:
			return cgenSceneXMLMetadata{}, errors.New("scene XML directives are not allowed")
		case xml.StartElement:
			seenAttributes := make(map[xml.Name]struct{}, len(typed.Attr))
			for _, attribute := range typed.Attr {
				if _, duplicate := seenAttributes[attribute.Name]; duplicate {
					return cgenSceneXMLMetadata{}, errors.New("scene XML contains duplicate attributes")
				}
				seenAttributes[attribute.Name] = struct{}{}
			}
			if depth == 0 {
				if rootSeen {
					return cgenSceneXMLMetadata{}, errors.New("scene XML is invalid")
				}
				rootSeen = true
				metadata.XMLName = typed.Name
				for _, attribute := range typed.Attr {
					if attribute.Name.Space != "" {
						continue
					}
					switch attribute.Name.Local {
					case "schema_version":
						metadata.SchemaVersion = attribute.Value
					case "id":
						metadata.ID = attribute.Value
					case "name":
						metadata.Name = attribute.Value
					}
				}
			}
			if typed.Name.Space == "" && typed.Name.Local == "node" {
				if nodeDepth == 0 {
					rootNodeCount++
				}
				nodeDepth++
				nodeCount++
				if nodeDepth > cgenSceneMaxNodeDepth {
					return cgenSceneXMLMetadata{}, errors.New("scene XML exceeds the node depth limit")
				}
				if nodeCount > cgenSceneMaxNodeCount {
					return cgenSceneXMLMetadata{}, errors.New("scene XML exceeds the node count limit")
				}
				var nodeID string
				for _, attribute := range typed.Attr {
					if attribute.Name.Space == "" && attribute.Name.Local == "id" {
						nodeID = attribute.Value
						break
					}
				}
				if !validCgenSceneIdentifierSyntax(nodeID) {
					return cgenSceneXMLMetadata{}, errors.New("scene XML contains an invalid node id")
				}
				if _, duplicate := nodeIDs[nodeID]; duplicate {
					return cgenSceneXMLMetadata{}, errors.New("scene XML contains duplicate node ids")
				}
				nodeIDs[nodeID] = struct{}{}
			}
			if typed.Name.Space == "" && typed.Name.Local == "image" {
				var assetID string
				for _, attribute := range typed.Attr {
					if attribute.Name.Space == "" && attribute.Name.Local == "asset" {
						assetID = attribute.Value
						break
					}
				}
				if !validCgenSceneAssetID(assetID) {
					return cgenSceneXMLMetadata{}, errors.New("scene XML contains an invalid image asset id")
				}
			}
			depth++
			if depth > cgenSceneMaxXMLDepth {
				return cgenSceneXMLMetadata{}, errors.New("scene XML exceeds the element depth limit")
			}
		case xml.EndElement:
			if typed.Name.Space == "" && typed.Name.Local == "node" {
				nodeDepth--
				if nodeDepth < 0 {
					return cgenSceneXMLMetadata{}, errors.New("scene XML is invalid")
				}
			}
			depth--
			if depth < 0 {
				return cgenSceneXMLMetadata{}, errors.New("scene XML is invalid")
			}
		case xml.CharData:
			if depth == 0 && strings.TrimSpace(string(typed)) != "" {
				return cgenSceneXMLMetadata{}, errors.New("scene XML is invalid")
			}
		}
	}
	if !rootSeen || depth != 0 || nodeDepth != 0 {
		return cgenSceneXMLMetadata{}, errors.New("scene XML is invalid")
	}
	if metadata.XMLName.Local != "scene" || metadata.XMLName.Space != "" {
		return cgenSceneXMLMetadata{}, errors.New("scene XML must have an unqualified scene root")
	}
	if strings.TrimSpace(metadata.SchemaVersion) != cgenSceneSchema {
		return cgenSceneXMLMetadata{}, errors.New("scene schema_version is unsupported")
	}
	if metadata.ID != strings.TrimSpace(metadata.ID) || !validCgenSceneID(metadata.ID) {
		return cgenSceneXMLMetadata{}, errors.New("scene id is invalid")
	}
	if metadata.Name != strings.TrimSpace(metadata.Name) || !validCgenSceneName(metadata.Name) {
		return cgenSceneXMLMetadata{}, errors.New("scene name is invalid")
	}
	if rootNodeCount != 1 {
		return cgenSceneXMLMetadata{}, errors.New("scene XML must contain exactly one root node")
	}
	return metadata, nil
}

func validCgenSceneAssetID(value string) bool {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 512 {
		return false
	}
	if strings.ContainsAny(value, "\\:\x00\r\n") || strings.HasPrefix(value, "/") {
		return false
	}
	parts := strings.Split(value, "/")
	for _, part := range parts {
		if part == "" || part == "." || part == ".." || isWindowsReservedCgenSceneComponent(part) {
			return false
		}
		for _, character := range part {
			if unicode.IsControl(character) {
				return false
			}
		}
	}
	return true
}

func isWindowsReservedCgenSceneComponent(value string) bool {
	value = strings.TrimSuffix(strings.TrimSpace(value), ".")
	stem := strings.ToUpper(strings.SplitN(value, ".", 2)[0])
	if stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL" || stem == "CLOCK$" {
		return true
	}
	if len(stem) == 4 && (strings.HasPrefix(stem, "COM") || strings.HasPrefix(stem, "LPT")) {
		return stem[3] >= '1' && stem[3] <= '9'
	}
	return false
}

func cgenSceneXMLPayload(payload map[string]any) ([]byte, error) {
	raw, ok := payload["xml"].(string)
	if !ok {
		raw, ok = payload["content"].(string)
	}
	if !ok || raw == "" {
		return nil, errors.New("scene XML is required")
	}
	if len(raw) > cgenSceneMaxXMLBytes {
		return nil, errors.New("scene XML exceeds the size limit")
	}
	return []byte(raw), nil
}

func requiredCgenSceneRevision(payload map[string]any) (string, error) {
	raw, exists := payload["expected_revision"]
	if !exists {
		return "", errors.New("expected_revision is required")
	}
	revision, ok := raw.(string)
	if !ok {
		return "", errors.New("expected_revision must be a string")
	}
	if revision != "" && !validCgenSceneRevision(revision) {
		return "", errors.New("expected_revision is invalid")
	}
	return revision, nil
}

func validCgenSceneRevision(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validCgenSceneID(value string) bool {
	if !validCgenSceneIdentifierSyntax(value) {
		return false
	}
	for _, protected := range cgenProtectedScenes {
		if strings.EqualFold(value, protected.ID) && value != protected.ID {
			return false
		}
	}
	return true
}

func validCgenSceneIdentifierSyntax(value string) bool {
	if len(value) == 0 || len(value) > 96 {
		return false
	}
	for index, char := range value {
		if char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' {
			continue
		}
		if index > 0 && (char == '_' || char == '-' || char == '.') {
			continue
		}
		return false
	}
	return true
}

func validCgenSceneFilename(value string) bool {
	if value == "" || len(value) > 100 || filepath.IsAbs(value) || filepath.Base(value) != value {
		return false
	}
	if strings.ContainsAny(value, `/\\`) || !strings.HasSuffix(strings.ToLower(value), ".xml") {
		return false
	}
	stem := value[:len(value)-len(".xml")]
	return validCgenSceneIdentifierSyntax(stem)
}

func validCgenSceneName(value string) bool {
	if value == "" || !utf8.ValidString(value) || utf8.RuneCountInString(value) > 160 {
		return false
	}
	for _, char := range value {
		if unicode.IsControl(char) {
			return false
		}
	}
	return true
}

func protectedCgenSceneByID(id string) (cgenProtectedScene, bool) {
	for _, scene := range cgenProtectedScenes {
		if scene.ID == id {
			return scene, true
		}
	}
	return cgenProtectedScene{}, false
}

func protectedCgenSceneByFilename(filename string) (cgenProtectedScene, bool) {
	for _, scene := range cgenProtectedScenes {
		if strings.EqualFold(scene.Filename, filename) {
			return scene, true
		}
	}
	return cgenProtectedScene{}, false
}

func cgenSceneTargetFilename(id string, requested string, source cgenSceneRecord, sourceFound bool) (string, error) {
	requested = strings.TrimSpace(requested)
	if requested != "" && !validCgenSceneFilename(requested) {
		return "", errors.New("scene filename is invalid")
	}
	if protected, ok := protectedCgenSceneByID(id); ok {
		if requested != "" && requested != protected.Filename {
			return "", errors.New("protected scenes cannot be renamed")
		}
		return protected.Filename, nil
	}
	if requested != "" {
		if sourceFound && strings.EqualFold(requested, source.Filename) {
			return source.Filename, nil
		}
		if _, reserved := protectedCgenSceneByFilename(requested); reserved {
			return "", errors.New("scene filename is reserved")
		}
		return requested, nil
	}
	if sourceFound && source.Metadata.ID == id {
		return source.Filename, nil
	}
	filename := id + ".xml"
	if !validCgenSceneFilename(filename) {
		return "", errors.New("scene filename is invalid")
	}
	if sourceFound && strings.EqualFold(filename, source.Filename) {
		return source.Filename, nil
	}
	return filename, nil
}

func validateCgenSceneRecordIdentity(record cgenSceneRecord) error {
	if protected, ok := protectedCgenSceneByID(record.Metadata.ID); ok {
		if record.Metadata.Name != protected.Name || record.Filename != protected.Filename {
			return errors.New("scene storage contains a renamed protected scene")
		}
		return nil
	}
	if _, reserved := protectedCgenSceneByFilename(record.Filename); reserved {
		return errors.New("scene storage contains an invalid protected filename")
	}
	return nil
}

func selectCgenSceneRecord(records []cgenSceneRecord, payload map[string]any) (cgenSceneRecord, bool, error) {
	id := strings.TrimSpace(firstNonBlank(stringValue(payload, "scene_id"), stringValue(payload, "id")))
	filename := strings.TrimSpace(stringValue(payload, "filename"))
	if id == "" && filename == "" {
		return cgenSceneRecord{}, false, errors.New("scene id or filename is required")
	}
	if id != "" && !validCgenSceneID(id) {
		return cgenSceneRecord{}, false, errors.New("scene id is invalid")
	}
	if filename != "" && !validCgenSceneFilename(filename) {
		return cgenSceneRecord{}, false, errors.New("scene filename is invalid")
	}
	var byID cgenSceneRecord
	var idFound bool
	if id != "" {
		byID, idFound = findCgenSceneRecordByID(records, id)
	}
	var byFilename cgenSceneRecord
	var filenameFound bool
	if filename != "" {
		byFilename, filenameFound = findCgenSceneRecordByFilename(records, filename)
	}
	if id != "" && filename != "" {
		if !idFound || !filenameFound {
			return cgenSceneRecord{}, false, nil
		}
		if byID.Filename != byFilename.Filename {
			return cgenSceneRecord{}, false, errors.New("scene id and filename do not identify the same scene")
		}
		return byID, true, nil
	}
	if id != "" {
		return byID, idFound, nil
	}
	return byFilename, filenameFound, nil
}

func findCgenSceneRecordByID(records []cgenSceneRecord, id string) (cgenSceneRecord, bool) {
	for _, record := range records {
		if record.Metadata.ID == id {
			return record, true
		}
	}
	return cgenSceneRecord{}, false
}

func findCgenSceneRecordByFilename(records []cgenSceneRecord, filename string) (cgenSceneRecord, bool) {
	for _, record := range records {
		if strings.EqualFold(record.Filename, filename) {
			return record, true
		}
	}
	return cgenSceneRecord{}, false
}

func cgenSceneRecordFromRaw(filename string, metadata cgenSceneXMLMetadata, raw []byte) cgenSceneRecord {
	protected, isProtected := protectedCgenSceneByID(metadata.ID)
	return cgenSceneRecord{
		Metadata:  metadata,
		Filename:  filename,
		Revision:  cgenSceneRevision(raw),
		XML:       raw,
		Protected: isProtected,
		Locked:    isProtected && protected.Locked,
	}
}

func cgenSceneRecordPayload(record cgenSceneRecord) map[string]any {
	return map[string]any{
		"id":             record.Metadata.ID,
		"name":           record.Metadata.Name,
		"filename":       record.Filename,
		"schema_version": record.Metadata.SchemaVersion,
		"revision":       record.Revision,
		"protected":      record.Protected,
		"locked":         record.Locked,
	}
}

func cgenSceneListPayload(records []cgenSceneRecord) map[string]any {
	scenes := make([]map[string]any, 0, len(records))
	for _, record := range records {
		scenes = append(scenes, cgenSceneRecordPayload(record))
	}
	return map[string]any{
		"scenes":   scenes,
		"revision": cgenSceneCollectionRevision(records),
	}
}

func cgenSceneRevision(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func cgenSceneCollectionRevision(records []cgenSceneRecord) string {
	hash := sha256.New()
	for _, record := range records {
		_, _ = io.WriteString(hash, record.Metadata.ID)
		_, _ = hash.Write([]byte{0})
		_, _ = io.WriteString(hash, record.Filename)
		_, _ = hash.Write([]byte{0})
		_, _ = io.WriteString(hash, record.Revision)
		_, _ = hash.Write([]byte{0})
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func writeCgenSceneAtomic(root *os.Root, filename string, raw []byte) error {
	if !validCgenSceneFilename(filename) {
		return errors.New("scene filename is invalid")
	}
	if info, err := root.Lstat(filename); err == nil {
		if !info.Mode().IsRegular() {
			return errors.New("scene target is not a regular file")
		}
	} else if !os.IsNotExist(err) {
		return errors.New("scene target could not be checked")
	}

	tempName, file, err := createCgenSceneTemp(root)
	if err != nil {
		return err
	}
	tempPresent := true
	defer func() {
		if tempPresent {
			_ = root.Remove(tempName)
		}
	}()

	if _, err := file.Write(raw); err != nil {
		_ = file.Close()
		return errors.New("scene could not be written")
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return errors.New("scene could not be written")
	}
	if err := file.Close(); err != nil {
		return errors.New("scene could not be written")
	}
	if err := root.Rename(tempName, filename); err != nil {
		return errors.New("scene could not be saved atomically")
	}
	tempPresent = false
	if err := root.Chmod(filename, 0o600); err != nil {
		return errors.New("scene permissions could not be secured")
	}
	return nil
}

func createCgenSceneTemp(root *os.Root) (string, *os.File, error) {
	for attempt := 0; attempt < 8; attempt++ {
		var random [12]byte
		if _, err := rand.Read(random[:]); err != nil {
			return "", nil, errors.New("scene temporary file could not be created")
		}
		name := fmt.Sprintf(".scene-%x.tmp", random[:])
		file, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err == nil {
			return name, file, nil
		}
		if !os.IsExist(err) {
			return "", nil, errors.New("scene temporary file could not be created")
		}
	}
	return "", nil, errors.New("scene temporary file could not be created")
}
