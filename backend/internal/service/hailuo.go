package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// MiniMax (Hailuo / 海螺) video generation endpoints.
//
//	POST {base}/v1/video_generation?GroupId={group_id}
//	GET  {base}/v1/query/video_generation?task_id={id}&GroupId={group_id}
//	GET  {base}/v1/files/retrieve?file_id={id}&GroupId={group_id}
//
// The third hop turns MiniMax's file_id into a downloadable URL; the status
// response is rewritten with that URL so downstream clients get a playable link.
const (
	HailuoEndpointCreate GrokMediaEndpoint        = "hailuo_create"
	HailuoEndpointStatus GrokMediaEndpoint        = "hailuo_status"
	OpenAIEndpointCapabilityHailuo OpenAIEndpointCapability = "hailuo"
	hailuoTaskKeyPrefix                            = "hailuo:"
	hailuoSubmitPath                               = "/v1/video_generation"
	hailuoQueryPath                                = "/v1/query/video_generation"
	hailuoRetrievePath                             = "/v1/files/retrieve"
)

func (e GrokMediaEndpoint) IsHailuo() bool {
	return e == HailuoEndpointCreate || e == HailuoEndpointStatus
}

// HailuoTaskKey isolates ownership and billing keys from other video providers.
func HailuoTaskKey(id string) string { return hailuoTaskKeyPrefix + strings.TrimSpace(id) }

// HailuoRequestInfo carries create-time billing facts.
type HailuoRequestInfo struct {
	Model           string
	Prompt          string
	DurationSeconds int
	VideoResolution string
}

// ParseHailuoRequest reads a MiniMax video_generation request body.
func ParseHailuoRequest(body []byte) (HailuoRequestInfo, error) {
	var info HailuoRequestInfo
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return info, fmt.Errorf("request body must be a JSON object")
	}
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	if model == "" {
		return info, fmt.Errorf("model is required")
	}
	info.Model = model
	info.Prompt = gjson.GetBytes(body, "prompt").String()
	if d := gjson.GetBytes(body, "duration"); d.Exists() {
		switch d.Type {
		case gjson.Number:
			info.DurationSeconds = int(d.Int())
		default:
			// MiniMax also accepts the string form "6".
			var n int
			if _, err := fmt.Sscanf(strings.TrimSpace(d.String()), "%d", &n); err == nil {
				info.DurationSeconds = n
			}
		}
	}
	info.VideoResolution = hailuoResolution(gjson.GetBytes(body, "resolution").String())
	return info, nil
}

func hailuoResolution(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	switch {
	case s == "":
		return ""
	case strings.Contains(s, "1080"):
		return "1080p"
	case strings.Contains(s, "768"), strings.Contains(s, "720"):
		return "720p"
	case strings.Contains(s, "512"), strings.Contains(s, "480"):
		return "480p"
	}
	return ""
}

// hailuoBaseURL strips any trailing path so callers may store either the origin
// or the documented "https://api.minimax.chat/v1" base.
func hailuoBaseURL(raw string) string {
	base := strings.TrimRight(strings.TrimSpace(raw), "/")
	base = strings.TrimSuffix(base, "/v1")
	return base
}

func buildHailuoURL(base, groupID string, endpoint GrokMediaEndpoint, taskID string) (string, error) {
	root := hailuoBaseURL(base)
	if root == "" {
		return "", fmt.Errorf("hailuo account missing base_url")
	}
	q := url.Values{}
	if strings.TrimSpace(groupID) != "" {
		q.Set("GroupId", strings.TrimSpace(groupID))
	}
	switch endpoint {
	case HailuoEndpointCreate:
		return root + hailuoSubmitPath + "?" + q.Encode(), nil
	default:
		if err := validateUpstreamPathSegment("Hailuo task ID", taskID); err != nil || strings.TrimSpace(taskID) == "" {
			return "", fmt.Errorf("invalid Hailuo task ID")
		}
		q.Set("task_id", taskID)
		return root + hailuoQueryPath + "?" + q.Encode(), nil
	}
}

// ForwardHailuo proxies the MiniMax native async video protocol. On status
// Success it resolves file_id -> download_url so the client always receives a
// playable URL, mirroring the behavior of the other native video providers.
func (s *OpenAIGatewayService) ForwardHailuo(ctx context.Context, c *gin.Context, account *Account, endpoint GrokMediaEndpoint, taskID string, body []byte) (*OpenAIForwardResult, error) {
	if !account.SupportsOpenAIEndpointCapability(OpenAIEndpointCapabilityHailuo) || !endpoint.IsHailuo() {
		return nil, fmt.Errorf("hailuo requires an OpenAI API key account with a custom base URL and the hailuo capability")
	}
	base, err := s.validateUpstreamBaseURL(account.GetCredential("base_url"))
	if err != nil {
		return nil, err
	}
	groupID := account.GetCredential("group_id")
	rawTaskID := strings.TrimPrefix(taskID, hailuoTaskKeyPrefix)
	target, err := buildHailuoURL(base, groupID, endpoint, rawTaskID)
	if err != nil {
		return nil, err
	}

	model, upstreamModel := "", ""
	method := http.MethodGet
	info := HailuoRequestInfo{}
	switch endpoint {
	case HailuoEndpointCreate:
		info, err = ParseHailuoRequest(body)
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
		return nil, fmt.Errorf("hailuo account missing api_key")
	}
	started := time.Now()
	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
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
	if resp.StatusCode >= 300 {
		writeGrokMediaResponse(c, resp, responseBody, s.responseHeaderFilter)
		return nil, fmt.Errorf("hailuo upstream status %d", resp.StatusCode)
	}

	result := &OpenAIForwardResult{
		Model:           model,
		BillingModel:    model,
		UpstreamModel:   upstreamModel,
		Duration:        time.Since(started),
		ResponseHeaders: resp.Header.Clone(),
	}

	switch endpoint {
	case HailuoEndpointCreate:
		id := strings.TrimSpace(gjson.GetBytes(responseBody, "task_id").String())
		if id == "" {
			return nil, fmt.Errorf("hailuo create response missing task_id")
		}
		result.ResponseID = HailuoTaskKey(id)
		result.VideoResolution = info.VideoResolution
		result.VideoDurationSeconds = info.DurationSeconds
	default:
		result.ResponseID = taskID
		status := strings.ToLower(strings.TrimSpace(gjson.GetBytes(responseBody, "status").String()))
		if status == "success" {
			fileID := strings.TrimSpace(gjson.GetBytes(responseBody, "file_id").String())
			if fileID != "" {
				if downloadURL, dlErr := s.hailuoFileDownloadURL(ctx, c, account, base, groupID, fileID, proxy); dlErr == nil && downloadURL != "" {
					if patched, setErr := sjson.SetBytes(responseBody, "video_url", downloadURL); setErr == nil {
						responseBody = patched
					}
					result.VideoCount = 1
				}
			} else if strings.TrimSpace(gjson.GetBytes(responseBody, "video_url").String()) != "" {
				// Some deployments return the URL inline.
				result.VideoCount = 1
			}
			if res := hailuoResolution(gjson.GetBytes(responseBody, "resolution").String()); res != "" {
				result.VideoResolution = res
			}
			if d := gjson.GetBytes(responseBody, "duration"); d.Exists() {
				result.VideoDurationSeconds = int(d.Int())
			}
		}
	}

	writeGrokMediaResponse(c, resp, responseBody, s.responseHeaderFilter)
	return result, nil
}

// hailuoFileDownloadURL resolves MiniMax file_id to a temporary download URL.
func (s *OpenAIGatewayService) hailuoFileDownloadURL(ctx context.Context, c *gin.Context, account *Account, base, groupID, fileID, proxy string) (string, error) {
	token := strings.TrimSpace(account.GetCredential("api_key"))
	if token == "" {
		return "", fmt.Errorf("hailuo account missing api_key")
	}
	q := url.Values{}
	q.Set("file_id", fileID)
	if strings.TrimSpace(groupID) != "" {
		q.Set("GroupId", strings.TrimSpace(groupID))
	}
	target := hailuoBaseURL(base) + hailuoRetrievePath + "?" + q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")
	account.ApplyHeaderOverrides(req.Header)
	resp, err := s.httpUpstream.Do(req, proxy, account.ID, account.Concurrency)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	raw, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return "", err
	}
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("hailuo retrieve status %d", resp.StatusCode)
	}
	var payload struct {
		File struct {
			DownloadURL string `json:"download_url"`
		} `json:"file"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return "", err
	}
	return strings.TrimSpace(payload.File.DownloadURL), nil
}
