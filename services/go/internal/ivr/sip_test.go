package ivr

import (
	"context"
	"encoding/binary"
	"net"
	"strings"
	"testing"
)

func TestSIPInviteNegotiatesPCMUAndTelephoneEvents(t *testing.T) {
	service := &Service{cfg: loadedConfig{IVR: Config{RTP: rtpConfig{PortMin: 0, PortMax: 0}}, Prompts: defaultPromptConfig()}}
	request := parseSIPRequest(strings.Join([]string{
		"INVITE sip:haze@127.0.0.1 SIP/2.0",
		"Via: SIP/2.0/UDP 127.0.0.1:5062;branch=z9hG4bK-test",
		"From: <sip:caller@127.0.0.1>;tag=abc",
		"To: <sip:haze@127.0.0.1>",
		"Call-ID: call-1",
		"CSeq: 1 INVITE",
		"Contact: <sip:caller@127.0.0.1:5062>",
		"Content-Type: application/sdp",
		"Content-Length: 161",
		"",
		"v=0",
		"o=caller 0 0 IN IP4 127.0.0.1",
		"s=call",
		"c=IN IP4 127.0.0.1",
		"t=0 0",
		"m=audio 40000 RTP/AVP 0 101",
		"a=rtpmap:0 PCMU/8000",
		"a=rtpmap:101 telephone-event/8000",
		"",
	}, "\r\n"))

	call, response := service.acceptSIPInvite(context.Background(), request, &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 5062}, &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 5060})
	if call == nil {
		t.Fatalf("expected accepted call, response: %s", response)
	}
	call.close()
	if !strings.Contains(response, "SIP/2.0 200 OK") {
		t.Fatalf("response was not 200 OK: %s", response)
	}
	if !strings.Contains(response, "Content-Type: application/sdp") || !strings.Contains(response, "m=audio ") {
		t.Fatalf("response did not include SDP: %s", response)
	}
	if !strings.Contains(response, "a=rtpmap:0 PCMU/8000") || !strings.Contains(response, "telephone-event/8000") {
		t.Fatalf("response did not negotiate PCMU and telephone events: %s", response)
	}
}

func TestRTPDTMFDigitDecodesEndEvent(t *testing.T) {
	packet := make([]byte, 16)
	packet[0] = 0x80
	packet[1] = sipDefaultDTMFPayload
	binary.BigEndian.PutUint32(packet[4:8], 1234)
	packet[12] = 1
	packet[13] = 0x80
	packet[14] = 0
	packet[15] = 160

	digit, key := rtpDTMFDigit(packet, sipDefaultDTMFPayload)
	if digit != "1" || key == "" {
		t.Fatalf("digit=%q key=%q", digit, key)
	}
}
