package handler

import (
	"mime"
	"net/http"

	middleware "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

// WanVideoTasks exposes Alibaba Bailian (DashScope) native asynchronous video
// task protocol: POST /api/v1/services/aigc/video-generation/video-synthesis
// and GET /api/v1/tasks/:task_id.
func (h *OpenAIGatewayHandler) WanVideoTasks(c *gin.Context) {
	if c.Request.Method == http.MethodPost && c.GetHeader("Content-Type") != "" {
		mediaType, _, err := mime.ParseMediaType(c.GetHeader("Content-Type"))
		if err != nil || mediaType != "application/json" {
			h.errorResponse(c, http.StatusUnsupportedMediaType, "invalid_request_error", "Wan video requires application/json")
			return
		}
	}
	key, ok := middleware.GetAPIKeyFromContext(c)
	if !ok || key.Group == nil || (key.Group.Platform != service.PlatformOpenAI && key.Group.Platform != service.PlatformComposite) {
		h.errorResponse(c, http.StatusForbidden, "permission_error", "Wan video requires an OpenAI or composite group")
		return
	}
	endpoint := service.WanEndpointCreate
	setActualUpstreamEndpoint(c, EndpointWanVideoSynthesis)
	taskID := ""
	if c.Request.Method != http.MethodPost {
		taskID = service.WanTaskKey(c.Param("task_id"))
		endpoint = service.WanEndpointStatus
	}
	h.handleGrokMedia(c, endpoint, taskID)
}
