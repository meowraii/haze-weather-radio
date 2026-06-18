package ivr

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	sipPayloadPCMU         = 0
	sipDefaultDTMFPayload  = 101
	sipTelephoneEventClock = 8000
	sipPacketSamples       = 160
)

type sipRequest struct {
	Method  string
	URI     string
	Headers map[string]string
	Body    string
}

type sipMediaOffer struct {
	Host        string
	Port        int
	DTMFPayload int
}

type sipCall struct {
	service     *Service
	ctx         context.Context
	cancel      context.CancelFunc
	callID      string
	localTag    string
	rtpConn     *net.UDPConn
	remoteRTP   *net.UDPAddr
	dtmfPayload int
	digits      chan string
	done        chan struct{}
	sendMu      sync.Mutex
	seq         uint16
	timestamp   uint32
	ssrc        uint32
}

func (s *Service) runSIP(ctx context.Context) error {
	addr, err := net.ResolveUDPAddr("udp", s.cfg.IVR.SIP.Listen)
	if err != nil {
		return err
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return err
	}
	defer conn.Close()
	log.Printf("IVR SIP listening on %s", s.cfg.IVR.SIP.Listen)

	go func() {
		<-ctx.Done()
		_ = conn.Close()
	}()

	calls := map[string]*sipCall{}
	var callsMu sync.Mutex
	buffer := make([]byte, 16384)
	for {
		n, remote, err := conn.ReadFromUDP(buffer)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		s.metrics.SIPMessages.Add(1)
		request := parseSIPRequest(string(buffer[:n]))
		if request.Method == "" {
			continue
		}
		callID := sipHeader(request.Headers, "call-id")
		switch request.Method {
		case "OPTIONS":
			_, _ = conn.WriteToUDP([]byte(sipReply("200 OK", request.Headers, "Allow: INVITE, ACK, BYE, CANCEL, OPTIONS, INFO\r\nAccept: application/sdp\r\n")), remote)
		case "INVITE":
			call, response := s.acceptSIPInvite(ctx, request, remote, conn.LocalAddr())
			_, _ = conn.WriteToUDP([]byte(response), remote)
			if call == nil || callID == "" {
				continue
			}
			callsMu.Lock()
			if existing := calls[callID]; existing != nil {
				existing.close()
			}
			calls[callID] = call
			callsMu.Unlock()
			go func(id string, c *sipCall) {
				c.run()
				callsMu.Lock()
				if calls[id] == c {
					delete(calls, id)
				}
				callsMu.Unlock()
			}(callID, call)
		case "ACK":
		case "BYE", "CANCEL":
			callsMu.Lock()
			call := calls[callID]
			delete(calls, callID)
			callsMu.Unlock()
			if call != nil {
				call.close()
			}
			_, _ = conn.WriteToUDP([]byte(sipReply("200 OK", request.Headers, "")), remote)
		case "INFO":
			if digit := sipInfoDigit(request.Body); digit != "" {
				callsMu.Lock()
				call := calls[callID]
				callsMu.Unlock()
				if call != nil {
					call.pushDigit(digit)
				}
			}
			_, _ = conn.WriteToUDP([]byte(sipReply("200 OK", request.Headers, "")), remote)
		default:
			_, _ = conn.WriteToUDP([]byte(sipReply("501 Not Implemented", request.Headers, "")), remote)
		}
	}
}

func (s *Service) acceptSIPInvite(ctx context.Context, request sipRequest, remote *net.UDPAddr, local net.Addr) (*sipCall, string) {
	offer, err := parseSDPOffer(request.Body, remote)
	if err != nil {
		return nil, sipReply("488 Not Acceptable Here", request.Headers, "Warning: 399 haze \""+escapeSIPWarning(err.Error())+"\"\r\n")
	}
	rtpConn, rtpPort, err := listenRTPPort(s.cfg.IVR.RTP.PortMin, s.cfg.IVR.RTP.PortMax)
	if err != nil {
		return nil, sipReply("503 Service Unavailable", request.Headers, "Warning: 399 haze \"no RTP ports available\"\r\n")
	}
	callCtx, cancel := context.WithCancel(ctx)
	localHost := s.sipAdvertiseHost(remote, local)
	call := &sipCall{
		service:     s,
		ctx:         callCtx,
		cancel:      cancel,
		callID:      sipHeader(request.Headers, "call-id"),
		localTag:    randomHex(6),
		rtpConn:     rtpConn,
		remoteRTP:   &net.UDPAddr{IP: net.ParseIP(offer.Host), Port: offer.Port},
		dtmfPayload: offer.DTMFPayload,
		digits:      make(chan string, 16),
		done:        make(chan struct{}),
		seq:         uint16(time.Now().UnixNano()),
		timestamp:   uint32(time.Now().UnixNano()),
		ssrc:        randomUint32(),
	}
	if call.dtmfPayload <= 0 {
		call.dtmfPayload = sipDefaultDTMFPayload
	}
	sdp := sipAnswerSDP(localHost, rtpPort, call.dtmfPayload)
	return call, sipReplyWithBody("200 OK", request.Headers, call.localTag, "application/sdp", sdp, localHost)
}

func (s *Service) sipAdvertiseHost(remote *net.UDPAddr, local net.Addr) string {
	if host := strings.TrimSpace(s.cfg.IVR.SIP.PublicHost); host != "" {
		return host
	}
	if remote != nil {
		if conn, err := net.DialUDP("udp", nil, &net.UDPAddr{IP: remote.IP, Port: 9}); err == nil {
			defer conn.Close()
			if addr, ok := conn.LocalAddr().(*net.UDPAddr); ok && addr.IP != nil && !addr.IP.IsUnspecified() {
				return addr.IP.String()
			}
		}
	}
	if udp, ok := local.(*net.UDPAddr); ok && udp.IP != nil && !udp.IP.IsUnspecified() {
		return udp.IP.String()
	}
	return "127.0.0.1"
}

func (c *sipCall) run() {
	defer c.close()
	go c.readRTP()
	c.menuLoop()
}

func (c *sipCall) close() {
	select {
	case <-c.done:
		return
	default:
		close(c.done)
		c.cancel()
		_ = c.rtpConn.Close()
	}
}

func (c *sipCall) menuLoop() {
	language := c.service.cfg.IVR.DefaultLanguage
	for c.ctx.Err() == nil {
		entry, _ := c.service.cfg.Prompts.Menu("entry")
		c.playPrompt("entry", "main", nil)
		digit, ok := c.waitDigit(entry.Timeout)
		if !ok {
			c.playPrompt("error", "timeout", nil)
			return
		}
		option, ok := c.service.cfg.Prompts.Option("entry", digit)
		if !ok {
			c.playPrompt("error", "invalid_code", nil)
			continue
		}
		switch option.Action {
		case "language":
			language = fallbackText(option.Language, language)
			location, ok := c.collectLocation(language)
			if !ok {
				return
			}
			c.locationMenu(location)
		case "product":
			location, err := c.service.defaultFeedLocation()
			if err != nil {
				c.playPrompt("weather_product", "unavailable", nil)
				return
			}
			if option.Language != "" {
				location.Language = option.Language
			}
			c.playProduct(location, splitCSV(option.Packages))
		case "broadcast":
			location, err := c.service.defaultFeedLocation()
			if err != nil {
				c.playPrompt("broadcast_menu", "main", nil)
				continue
			}
			location.Language = language
			c.playProduct(location, c.service.broadcastPackages(option))
		case "operator":
			c.playPrompt("operator", "main", nil)
			return
		default:
			c.playPrompt("error", "invalid_code", nil)
		}
	}
}

func (c *sipCall) collectLocation(language string) (ResolvedLocation, bool) {
	menu, _ := c.service.cfg.Prompts.Menu("location_code")
	attempts := maxInt(1, menu.Retries+1)
	for attempt := 0; attempt < attempts && c.ctx.Err() == nil; attempt++ {
		c.playPrompt("location_code", "main", nil)
		code, ok := c.collectDigits(menu.Timeout)
		if !ok {
			c.playPrompt("error", "timeout", nil)
			return ResolvedLocation{}, false
		}
		location, err := c.service.resolver.Resolve(code)
		if err == nil {
			location.Language = language
			return location, true
		}
		c.playPrompt("error", "invalid_code", nil)
	}
	return ResolvedLocation{}, false
}

func (c *sipCall) locationMenu(location ResolvedLocation) {
	menu, _ := c.service.cfg.Prompts.Menu("location_menu")
	for c.ctx.Err() == nil {
		c.playPrompt("location_menu", "main", map[string]string{"location": location.Name})
		digit, ok := c.waitDigit(menu.Timeout)
		if !ok {
			c.playPrompt("error", "timeout", nil)
			return
		}
		option, ok := c.service.cfg.Prompts.Option("location_menu", digit)
		if !ok {
			c.playPrompt("error", "invalid_code", nil)
			continue
		}
		switch option.Action {
		case "product":
			c.playProduct(location, splitCSV(option.Packages))
		case "broadcast":
			c.playProduct(location, c.service.broadcastPackages(option))
		case "operator":
			c.playPrompt("operator", "main", nil)
			return
		default:
			c.playPrompt("error", "invalid_code", nil)
		}
	}
}

func (c *sipCall) playPrompt(menuID string, lineKey string, values map[string]string) {
	audio, err := c.service.cache.GetPrompt(c.ctx, menuID, lineKey, values, false)
	if err != nil {
		log.Printf("IVR SIP prompt %s/%s failed: %v", menuID, lineKey, err)
		return
	}
	c.playPCMUFile(audio.PCMUPath)
}

func (c *sipCall) playProduct(location ResolvedLocation, packages []string) {
	packages = normalizePackages(packages, c.service.cfg.IVR.DefaultPackages)
	if _, ok := c.service.cache.Fresh(location, packages); !ok {
		c.playPrompt("", "one_moment", nil)
	}
	type result struct {
		product CachedProduct
		err     error
	}
	done := make(chan result, 1)
	go func() {
		product, err := c.service.productForLocation(c.ctx, location, packages, false)
		done <- result{product: product, err: err}
	}()
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	silence := make([]byte, sipPacketSamples)
	for index := range silence {
		silence[index] = 0xFF
	}
	for {
		select {
		case <-c.ctx.Done():
			return
		case result := <-done:
			if result.err != nil {
				log.Printf("IVR SIP product unavailable for %s packages=%s: %v", firstNonBlank(location.Code, location.FeedID), strings.Join(packages, ","), result.err)
				c.playPrompt("weather_product", "unavailable", nil)
				return
			}
			c.playPCMUFile(result.product.PCMUPath)
			return
		case <-ticker.C:
			c.sendRTP(silence)
		}
	}
}

func (c *sipCall) playPCMUFile(path string) {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Printf("IVR SIP audio read failed: %v", err)
		return
	}
	if len(raw) == 0 {
		return
	}
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for offset := 0; offset < len(raw) && c.ctx.Err() == nil; offset += sipPacketSamples {
		end := offset + sipPacketSamples
		frame := make([]byte, sipPacketSamples)
		for index := range frame {
			frame[index] = 0xFF
		}
		if end > len(raw) {
			end = len(raw)
		}
		copy(frame, raw[offset:end])
		c.sendRTP(frame)
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (c *sipCall) sendRTP(payload []byte) {
	if c.remoteRTP == nil || c.rtpConn == nil || len(payload) == 0 {
		return
	}
	c.sendMu.Lock()
	defer c.sendMu.Unlock()
	packet := make([]byte, 12+len(payload))
	packet[0] = 0x80
	packet[1] = sipPayloadPCMU
	binary.BigEndian.PutUint16(packet[2:4], c.seq)
	binary.BigEndian.PutUint32(packet[4:8], c.timestamp)
	binary.BigEndian.PutUint32(packet[8:12], c.ssrc)
	copy(packet[12:], payload)
	_, _ = c.rtpConn.WriteToUDP(packet, c.remoteRTP)
	c.seq++
	c.timestamp += uint32(len(payload))
}

func (c *sipCall) readRTP() {
	buffer := make([]byte, 1500)
	var lastEvent string
	for {
		n, _, err := c.rtpConn.ReadFromUDP(buffer)
		if err != nil {
			return
		}
		digit, key := rtpDTMFDigit(buffer[:n], c.dtmfPayload)
		if digit == "" || key == lastEvent {
			continue
		}
		lastEvent = key
		c.pushDigit(digit)
	}
}

func (c *sipCall) pushDigit(digit string) {
	select {
	case c.digits <- digit:
	default:
	}
}

func (c *sipCall) waitDigit(timeout time.Duration) (string, bool) {
	if timeout <= 0 {
		timeout = time.Duration(c.service.cfg.IVR.DigitTimeoutSeconds) * time.Second
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case <-c.ctx.Done():
			return "", false
		case digit := <-c.digits:
			if digit != "" {
				return digit, true
			}
		case <-timer.C:
			return "", false
		}
	}
}

func (c *sipCall) collectDigits(timeout time.Duration) (string, bool) {
	if timeout <= 0 {
		timeout = time.Duration(c.service.cfg.IVR.DigitTimeoutSeconds) * time.Second
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	var builder strings.Builder
	for {
		select {
		case <-c.ctx.Done():
			return "", false
		case digit := <-c.digits:
			if digit == "#" {
				return builder.String(), builder.Len() > 0
			}
			if digit == "*" {
				builder.Reset()
				continue
			}
			if len(digit) == 1 && digit[0] >= '0' && digit[0] <= '9' {
				builder.WriteString(digit)
			}
			if builder.Len() >= 12 {
				return builder.String(), true
			}
		case <-timer.C:
			return builder.String(), builder.Len() > 0
		}
	}
}

func parseSIPRequest(raw string) sipRequest {
	normalized := strings.ReplaceAll(raw, "\r\n", "\n")
	head, body, _ := strings.Cut(normalized, "\n\n")
	lines := strings.Split(head, "\n")
	if len(lines) == 0 {
		return sipRequest{}
	}
	fields := strings.Fields(strings.TrimSpace(lines[0]))
	if len(fields) < 1 {
		return sipRequest{}
	}
	headers := sipHeaders(lines[1:])
	method := strings.ToUpper(fields[0])
	uri := ""
	if len(fields) > 1 {
		uri = fields[1]
	}
	return sipRequest{Method: method, URI: uri, Headers: headers, Body: body}
}

func parseSDPOffer(body string, remote *net.UDPAddr) (sipMediaOffer, error) {
	offer := sipMediaOffer{DTMFPayload: sipDefaultDTMFPayload}
	if remote != nil {
		offer.Host = remote.IP.String()
	}
	for _, line := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "c=IN IP4 ") {
			offer.Host = strings.TrimSpace(strings.TrimPrefix(line, "c=IN IP4 "))
			continue
		}
		if strings.HasPrefix(line, "m=audio ") {
			fields := strings.Fields(strings.TrimPrefix(line, "m=audio "))
			if len(fields) > 0 {
				port, _ := strconv.Atoi(fields[0])
				offer.Port = port
			}
			continue
		}
		lowerLine := strings.ToLower(line)
		if strings.HasPrefix(lowerLine, "a=rtpmap:") && strings.Contains(lowerLine, "telephone-event/8000") {
			payload := strings.TrimPrefix(lowerLine, "a=rtpmap:")
			payload = strings.Fields(payload)[0]
			number, _ := strconv.Atoi(payload)
			if number > 0 {
				offer.DTMFPayload = number
			}
		}
	}
	if net.ParseIP(offer.Host) == nil {
		return sipMediaOffer{}, fmt.Errorf("missing remote RTP host")
	}
	if offer.Port <= 0 || offer.Port > math.MaxUint16 {
		return sipMediaOffer{}, fmt.Errorf("missing remote RTP port")
	}
	return offer, nil
}

func listenRTPPort(minPort int, maxPort int) (*net.UDPConn, int, error) {
	if minPort <= 0 || maxPort < minPort {
		conn, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4zero, Port: 0})
		if err != nil {
			return nil, 0, err
		}
		return conn, conn.LocalAddr().(*net.UDPAddr).Port, nil
	}
	for port := minPort; port <= maxPort; port++ {
		conn, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4zero, Port: port})
		if err == nil {
			return conn, port, nil
		}
	}
	return nil, 0, fmt.Errorf("no free RTP port in %d-%d", minPort, maxPort)
}

func sipAnswerSDP(host string, port int, dtmfPayload int) string {
	return strings.Join([]string{
		"v=0",
		"o=haze 0 0 IN IP4 " + host,
		"s=Haze IVR",
		"c=IN IP4 " + host,
		"t=0 0",
		fmt.Sprintf("m=audio %d RTP/AVP 0 %d", port, dtmfPayload),
		"a=rtpmap:0 PCMU/8000",
		fmt.Sprintf("a=rtpmap:%d telephone-event/8000", dtmfPayload),
		fmt.Sprintf("a=fmtp:%d 0-15", dtmfPayload),
		"a=sendrecv",
		"",
	}, "\r\n")
}

func sipHeaders(lines []string) map[string]string {
	headers := map[string]string{}
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			break
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		headers[strings.ToLower(strings.TrimSpace(key))] = strings.TrimSpace(value)
	}
	return headers
}

func sipHeader(headers map[string]string, key string) string {
	return headers[strings.ToLower(strings.TrimSpace(key))]
}

func sipReply(status string, headers map[string]string, extra string) string {
	return sipBuildReply(status, headers, "", "", "", "", extra)
}

func sipReplyWithBody(status string, headers map[string]string, toTag string, contentType string, body string, contactHost string) string {
	return sipBuildReply(status, headers, toTag, contentType, body, contactHost, "")
}

func sipBuildReply(status string, headers map[string]string, toTag string, contentType string, body string, contactHost string, extra string) string {
	var builder strings.Builder
	builder.WriteString("SIP/2.0 " + status + "\r\n")
	for _, key := range []string{"via", "from", "to", "call-id", "cseq"} {
		if value := sipHeader(headers, key); value != "" {
			name := map[string]string{"call-id": "Call-ID", "cseq": "CSeq"}[key]
			if name == "" {
				name = strings.ToUpper(key[:1]) + key[1:]
			}
			if key == "to" && toTag != "" && !strings.Contains(strings.ToLower(value), "tag=") {
				value += ";tag=" + toTag
			}
			builder.WriteString(name + ": " + value + "\r\n")
		}
	}
	if contactHost != "" {
		builder.WriteString("Contact: <sip:haze@" + contactHost + ">\r\n")
	}
	builder.WriteString("Allow: INVITE, ACK, BYE, CANCEL, OPTIONS, INFO\r\n")
	builder.WriteString(extra)
	if contentType != "" && body != "" {
		builder.WriteString("Content-Type: " + contentType + "\r\n")
		builder.WriteString("Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n")
		builder.WriteString(body)
		return builder.String()
	}
	builder.WriteString("Content-Length: 0\r\n\r\n")
	return builder.String()
}

func sipInfoDigit(body string) string {
	for _, line := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(key), "Signal") {
			return normalizeDTMFDigit(strings.TrimSpace(value))
		}
	}
	return ""
}

func rtpDTMFDigit(packet []byte, payloadType int) (string, string) {
	if len(packet) < 16 {
		return "", ""
	}
	if int(packet[1]&0x7F) != payloadType {
		return "", ""
	}
	cc := int(packet[0] & 0x0F)
	offset := 12 + cc*4
	if len(packet) < offset+4 {
		return "", ""
	}
	event := int(packet[offset])
	end := packet[offset+1]&0x80 != 0
	if !end {
		return "", ""
	}
	timestamp := binary.BigEndian.Uint32(packet[4:8])
	digit := normalizeDTMFDigit(strconv.Itoa(event))
	if digit == "" {
		return "", ""
	}
	return digit, fmt.Sprintf("%d:%d", event, timestamp)
}

func normalizeDTMFDigit(value string) string {
	value = strings.TrimSpace(value)
	switch value {
	case "0", "1", "2", "3", "4", "5", "6", "7", "8", "9":
		return value
	case "10", "*":
		return "*"
	case "11", "#":
		return "#"
	default:
		return ""
	}
}

func randomHex(bytesCount int) string {
	buf := make([]byte, bytesCount)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x", buf)
}

func randomUint32() uint32 {
	var buf [4]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return uint32(time.Now().UnixNano())
	}
	return binary.BigEndian.Uint32(buf[:])
}

func escapeSIPWarning(value string) string {
	value = strings.ReplaceAll(value, `"`, `'`)
	return strings.ReplaceAll(value, "\r\n", " ")
}
