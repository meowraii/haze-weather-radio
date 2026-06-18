package webgateway

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const defaultDictionaryFile = "managed/dictionary.json"

type dictionaryPayload struct {
	Path      string                       `json:"path"`
	UpdatedAt string                       `json:"updated_at,omitempty"`
	Groups    map[string]map[string]string `json:"groups"`
	Summary   dictionarySummary            `json:"summary"`
}

type dictionarySummary struct {
	GroupCount int `json:"group_count"`
	EntryCount int `json:"entry_count"`
}

func loadDictionaryPayload(configPath string) (map[string]any, error) {
	path := dictionaryPath(configPath)
	groups, err := readDictionaryGroups(path)
	if err != nil {
		return nil, err
	}
	payload := dictionaryPayload{
		Path:    filepath.ToSlash(path),
		Groups:  groups,
		Summary: dictionarySummaryFor(groups),
	}
	if info, err := os.Stat(path); err == nil {
		payload.UpdatedAt = info.ModTime().UTC().Format("2006-01-02T15:04:05Z")
	}
	return dictionaryPayloadMap(payload), nil
}

func writeDictionaryPayload(configPath string, payload map[string]any) (map[string]any, error) {
	rawGroups, ok := payload["groups"]
	if !ok {
		return nil, fmt.Errorf("dictionary groups payload is required")
	}
	groups, err := normalizeDictionaryGroups(rawGroups)
	if err != nil {
		return nil, err
	}
	path := dictionaryPath(configPath)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	raw, err := json.MarshalIndent(groups, "", "  ")
	if err != nil {
		return nil, err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o600); err != nil {
		return nil, err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return nil, err
	}
	return loadDictionaryPayload(configPath)
}

func dictionaryPath(configPath string) string {
	return resolveConfigPath(configPath, defaultDictionaryFile)
}

func readDictionaryGroups(path string) (map[string]map[string]string, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]map[string]string{}, nil
		}
		return nil, err
	}
	var parsed map[string]map[string]string
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("dictionary json is invalid: %w", err)
	}
	return normalizeDictionaryGroups(parsed)
}

func normalizeDictionaryGroups(raw any) (map[string]map[string]string, error) {
	source, ok := raw.(map[string]map[string]string)
	if !ok {
		if generic, genericOK := raw.(map[string]any); genericOK {
			converted := make(map[string]map[string]string, len(generic))
			for group, entries := range generic {
				entryMap, entryOK := entries.(map[string]any)
				if !entryOK {
					return nil, fmt.Errorf("dictionary group %q must be an object", group)
				}
				converted[group] = make(map[string]string, len(entryMap))
				for match, replacement := range entryMap {
					replacementText, textOK := replacement.(string)
					if !textOK {
						return nil, fmt.Errorf("dictionary entry %q in %q must be text", match, group)
					}
					converted[group][match] = replacementText
				}
			}
			source = converted
			ok = true
		}
	}
	if !ok {
		return nil, fmt.Errorf("dictionary must be an object of language groups")
	}

	normalized := make(map[string]map[string]string, len(source))
	for group, entries := range source {
		group = strings.TrimSpace(group)
		if group == "" {
			return nil, fmt.Errorf("dictionary group names cannot be blank")
		}
		if len(group) > 48 {
			return nil, fmt.Errorf("dictionary group %q is too long", group)
		}
		if entries == nil {
			entries = map[string]string{}
		}
		cleanEntries := make(map[string]string, len(entries))
		for match, replacement := range entries {
			match = strings.TrimSpace(match)
			if match == "" {
				return nil, fmt.Errorf("dictionary entries in %q cannot have a blank match", group)
			}
			if len(match) > 200 {
				return nil, fmt.Errorf("dictionary entry %q in %q is too long", match, group)
			}
			replacement = strings.TrimSpace(replacement)
			if replacement == "" {
				continue
			}
			if len(replacement) > 400 {
				return nil, fmt.Errorf("dictionary replacement for %q in %q is too long", match, group)
			}
			cleanEntries[match] = replacement
		}
		normalized[group] = cleanEntries
	}
	return normalized, nil
}

func dictionarySummaryFor(groups map[string]map[string]string) dictionarySummary {
	summary := dictionarySummary{GroupCount: len(groups)}
	for _, entries := range groups {
		summary.EntryCount += len(entries)
	}
	return summary
}

func dictionaryPayloadMap(payload dictionaryPayload) map[string]any {
	groups := make(map[string]any, len(payload.Groups))
	groupNames := make([]string, 0, len(payload.Groups))
	for group := range payload.Groups {
		groupNames = append(groupNames, group)
	}
	sort.Strings(groupNames)
	for _, group := range groupNames {
		entries := payload.Groups[group]
		entryMap := make(map[string]any, len(entries))
		entryNames := make([]string, 0, len(entries))
		for match := range entries {
			entryNames = append(entryNames, match)
		}
		sort.Strings(entryNames)
		for _, match := range entryNames {
			entryMap[match] = entries[match]
		}
		groups[group] = entryMap
	}
	return map[string]any{
		"path":       payload.Path,
		"updated_at": payload.UpdatedAt,
		"groups":     groups,
		"summary": map[string]any{
			"group_count": payload.Summary.GroupCount,
			"entry_count": payload.Summary.EntryCount,
		},
	}
}
