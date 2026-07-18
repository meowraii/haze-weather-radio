package webgateway

import (
	"context"
	"database/sql"
	"testing"
)

func TestAccountPolicyPayloadAppliesOriginationRulesAndNormalizesSenderID(t *testing.T) {
	account := accountPolicyFromPayload(Account{Username: "operator"}, map[string]any{
		"allow_origination":         true,
		"allowed_originators":       []any{"eas", "WXR"},
		"blocked_event_codes":       []any{"ean", "NPT", "ean"},
		"force_originator_name":     true,
		"originator_name_text":      "Sherwood Weather",
		"include_ip_in_brackets":    true,
		"force_sender_id":           true,
		"sender_id":                 "ab-cd123",
		"allow_user_pw_change":      true,
		"password_expiry_days":      90,
		"logging_enabled":           true,
		"can_view_logs":             true,
		"allow_persistent_sessions": false,
	})
	if err := validateAccount(&account); err != nil {
		t.Fatal(err)
	}
	if !account.AllowOrigination || len(account.AllowedOriginators) != 2 {
		t.Fatalf("origination policy = allowed:%v originators:%v", account.AllowOrigination, account.AllowedOriginators)
	}
	if len(account.BlockedEventCodes) != 2 || account.BlockedEventCodes[0] != "EAN" || account.BlockedEventCodes[1] != "NPT" {
		t.Fatalf("blocked event codes = %v", account.BlockedEventCodes)
	}
	if account.SenderID != "AB/CD123" {
		t.Fatalf("sender ID = %q", account.SenderID)
	}
	if !account.ForceOriginatorName || account.OriginatorNameText != "Sherwood Weather" || !account.IncludeIPInBrackets {
		t.Fatalf("originator identity policy = %#v", account)
	}
}

func TestValidateAccountRejectsInvalidSenderIDCharacters(t *testing.T) {
	account := Account{Username: "operator", ForceSenderID: true, SenderID: "AB_CD123"}
	if err := validateAccount(&account); err == nil {
		t.Fatal("invalid sender ID was accepted")
	}
}

func TestValidateAccountRejectsInvalidBlockedEventCode(t *testing.T) {
	account := Account{Username: "operator", BlockedEventCodes: []string{"TOOLONG"}}
	if err := validateAccount(&account); err == nil {
		t.Fatal("invalid blocked event code was accepted")
	}
}

func TestEnsureAccountSecurityColumnsAddsBlockedEventPolicy(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE users (id TEXT PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}
	if err := ensureAccountSecurityColumns(context.Background(), db, "sqlite"); err != nil {
		t.Fatal(err)
	}
	rows, err := db.Query(`PRAGMA table_info(users)`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	found := false
	for rows.Next() {
		var cid, notNull, primaryKey int
		var name, columnType string
		var defaultValue any
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			t.Fatal(err)
		}
		found = found || name == "blocked_event_codes"
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("blocked_event_codes migration was not applied")
	}
}
