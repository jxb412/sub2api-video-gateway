package handler

import (
	"mime"
	"net/http"

	middleware "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

// HailuoVideoTasks exposes the MiniMax (海螺) native asynchronous video task
// protocol: POST /v1/video_generation and GET /v1/query/video_generation.
func (h *OpenAIGatewayHandler) HailuoVideoTasks(c *gin.Context) {
	if c.Request.Method == http.MethodPost && c.GetHeader("Content-Type") != "" {
		mediaType, _, err := mime.ParseMediaType(c.GetHeader("Content-Type"))
		if err != nil || mediaType != "application/json" {
			h.errorResponse(c, http.StatusUnsupportedMediaType, "invalid_request_error", "Hailuo video requires application/json")
			return
		}
	}
	key, ok := middleware.GetAPIKeyFromContext(c)
	if !ok || key.Group == nil || (key.Group.Platform != service.PlatformOpenAI && key.Group.Platform != service.PlatformComposite) {
		h.errorResponse(c, http.StatusForbidden, "permission_error", "Hailuo video requires an OpenAI or composite group")
		return
	}
	endpoint := service.HailuoEndpointCreate
	setActualUpstreamEndpoint(c, EndpointHailuoVideoGeneration)
	taskID := ""
	if c.Request.Method != http.MethodPost {
		taskID = service.HailuoTaskKey(c.Query("task_id"))
		endpoint = service.HailuoEndpointStatus
	}
	h.handleGrokMedia(c, endpoint, taskID)
}
