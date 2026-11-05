package service

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Alibaba Bailian (DashScope) video generation endpoints.
// Docs: POST {base}/api/v1/services/aigc/video-generation/video-synthesis
//
//	GET  {base}/api/v1/tasks/{task_id}
//
// The wire format is DashScope-native: {"model","input":{"prompt",...},"parameters":{"size","duration",...}}.
const (
	WanEndpointCreate                GrokMediaEndpoint        = "wan_create"
	WanEndpointStatus                GrokMediaEndpoint        = "wan_status"
	OpenAIEndpointCapabilityWan      OpenAIEndpointCapability = "wan"
	wanTaskKeyPrefix                                          = "wan:"
	wanVideoSynthesisPath                                     = "/api/v1/services/aigc/video-generation/video-synthesis"
	wanTasksPath                                              = "/api/v1/tasks"
)

func (e GrokMediaEndpoint) IsWan() bool {
	return e == WanEndpointCreate || e == WanEndpointStatus
}

// WanTaskKey isolates ownership and billing keys from other video providers.
func WanTaskKey(id string) string { return wanTaskKeyPrefix + strings.TrimSpace(id) }

// WanRequestInfo carries the fields needed for create-time billing snapshots.
type WanRequestInfo struct {
	Model            string
	Prompt           string
	ImageURL         string
	DurationSeconds  int
	VideoResolution  string
	HasInputImage    bool
	HasInputVideo    bool
}

// ParseWanRequest reads a DashScope-native video synthesis request body.
func ParseWanRequest(body []byte) (WanRequestInfo, error) {
	var info WanRequestInfo
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return info, fmt.Errorf("request body must be a JSON object")
	}
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	if model == "" {
		return info, fmt.Errorf("model is required")
	}
	info.Model = model
	info.Prompt = gjson.GetBytes(body, "input.prompt").String()
	info.ImageURL = strings.TrimSpace(gjson.GetBytes(body, "input.img_url").String())
	info.HasInputImage = info.ImageURL != ""
	info.HasInputVideo = strings.TrimSpace(gjson.GetBytes(body, "input.video_url").String()) != "" ||
		strings.TrimSpace(gjson.GetBytes(body, "input.ref_video_url").String()) != ""
	if d := gjson.GetBytes(body, "parameters.duration"); d.Exists() {
		info.DurationSeconds = int(d.Int())
	}
	info.VideoResolution = wanResolutionFromSize(gjson.GetBytes(body, "parameters.size").String())
	if info.VideoResolution == "" {
		info.VideoResolution = wanResolutionFromSize(gjson.GetBytes(body, "parameters.resolution").String())
	}
	return info, nil
}

// wanResolutionFromSize maps DashScope size/resolution values onto the billing
// resolution keys used by the shared video price table ("480p"/"720p"/"1080p").
func wanResolutionFromSize(size string) string {
	s := strings.ToLower(strings.TrimSpace(size))
	if s == "" {
		return ""
	}
	switch {
	case strings.Contains(s, "1080"):
		return "1080p"
	case strings.Contains(s, "720"):
		return "720p"
	case strings.Contains(s, "480"):
		return "480p"
	}
	if strings.Contains(s, "*") || strings.Contains(s, "x") {
		parts := strings.FieldsFunc(s, func(r rune) bool { return r == '*' || r == 'x' })
		if len(parts) == 2 {
			height := parts[1]
			switch {
			case strings.HasPrefix(height, "1080"):
				return "1080p"
			case strings.HasPrefix(height, "720"):
				return "720p"
			case strings.HasPrefix(height, "480"):
				return "480p"
			}
		}
	}
	return ""
}

// wanResolutionFromSR maps the DashScope usage.SR field (1080/720/480) to the
// billing resolution key.
func wanResolutionFromSR(sr int) string {
	switch {
	case sr >= 1080:
		return "1080p"
	case sr >= 720:
		return "720p"
	case sr > 0:
		return "480p"
	}
	return ""
}

// buildWanURL accepts an origin or a base URL that already contains /api/v1.
func buildWanURL(base string, endpoint GrokMediaEndpoint, taskID string) (string, error) {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if base == "" {
		return "", fmt.Errorf("wan account missing base_url")
	}
	switch {
	case strings.HasSuffix(base, wanVideoSynthesisPath), strings.HasSuffix(base, "/video-synthesis"):
		// already the full synthesis URL
	case strings.HasSuffix(base, "/api/v1"):
		base += "/services/aigc/video-generation/video-synthesis"
	default:
		base += wanVideoSynthesisPath
	}
	if endpoint == WanEndpointCreate {
		return base, nil
	}
	if err := validateUpstreamPathSegment("Wan task ID", taskID); err != nil || strings.TrimSpace(taskID) == "" {
		return "", fmt.Errorf("invalid Wan task ID")
	}
	origin := strings.SplitN(base, "/api/v1/", 2)[0]
	return origin + wanTasksPath + "/" + taskID, nil
}

// ForwardWan proxies the DashScope native asynchronous video task protocol.
// Only the model field is rewritten through the account model mapping; every
// other field is forwarded untouched so new DashScope parameters keep working.
func (s *OpenAIGatewayService) ForwardWan(ctx context.Context, c *gin.Context, account *Account, endpoint GrokMediaEndpoint, taskID string, body []byte) (*OpenAIForwardResult, error) {
	if !account.SupportsOpenAIEndpointCapability(OpenAIEndpointCapabilityWan) || !endpoint.IsWan() {
		return nil, fmt.Errorf("wan requires an OpenAI API key account with a custom base URL and the wan capability")
	}
	base, err := s.validateUpstreamBaseURL(account.GetCredential("base_url"))
	if err != nil {
		return nil, err
	}
	rawTaskID := strings.TrimPrefix(taskID, wanTaskKeyPrefix)
	target, err := buildWanURL(base, endpoint, rawTaskID)
	if err != nil {
		return nil, err
	}

	model, upstreamModel := "", ""
	method := http.MethodGet
	info := WanRequestInfo{}
	switch endpoint {
	case WanEndpointCreate:
		info, err = ParseWanRequest(body)
		if err != nil {
			return nil, err
		}
		model = info.Model
		upstreamModel = account.GetMappedModel(model)
		body, err = sjson.SetBytes(body, "model", upstreamModel)
		if err != nil {
			return nil, err
		}
		method = http.MethodPost
	}

	token := strings.TrimSpace(account.GetCredential("api_key"))
	if token == "" {
		return nil, fmt.Errorf("wan account missing api_key")
	}
	started := time.Now()
	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if endpoint == WanEndpointCreate {
		req.Header.Set("X-DashScope-Async", "enable")
	}
	account.ApplyHeaderOverrides(req.Header)
	proxy := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxy = account.Proxy.URL()
	}
	resp, err := s.httpUpstream.Do(req, proxy, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(started).Milliseconds())
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	responseBody, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return nil, err
	}
	// Never retry an ambiguous create: the upstream may already be billing.
	if resp.StatusCode >= 300 {
		writeGrokMediaResponse(c, resp, responseBody, s.responseHeaderFilter)
		return nil, fmt.Errorf("wan upstream status %d", resp.StatusCode)
	}

	result := &OpenAIForwardResult{
		Model:        model,
		BillingModel: model,
		UpstreamModel: upstreamModel,
		Duration:     time.Since(started),
		ResponseHeaders: resp.Header.Clone(),
	}
	switch endpoint {
	case WanEndpointCreate:
		id := strings.TrimSpace(gjson.GetBytes(responseBody, "output.task_id").String())
		if id == "" {
			return nil, fmt.Errorf("wan create response missing task_id")
		}
		result.ResponseID = WanTaskKey(id)
		result.VideoResolution = info.VideoResolution
		result.VideoDurationSeconds = info.DurationSeconds
	case WanEndpointStatus:
		result.ResponseID = taskID
		if m := strings.TrimSpace(gjson.GetBytes(responseBody, "output.model").String()); m != "" {
			result.UpstreamModel = m
		}
		status := strings.ToUpper(strings.TrimSpace(gjson.GetBytes(responseBody, "output.task_status").String()))
		videoURL := strings.TrimSpace(gjson.GetBytes(responseBody, "output.video_url").String())
		if status == "SUCCEEDED" && videoURL != "" {
			result.VideoCount = 1
			result.VideoResolution = wanResolutionFromSR(int(gjson.GetBytes(responseBody, "usage.SR").Int()))
			if d := gjson.GetBytes(responseBody, "usage.duration"); d.Exists() {
				result.VideoDurationSeconds = int(d.Float())
			}
		}
	}
	writeGrokMediaResponse(c, resp, responseBody, s.responseHeaderFilter)
	return result, nil
}
