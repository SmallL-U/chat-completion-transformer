package httpapi

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func requestLoggingMiddleware(logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		started := time.Now()
		c.Next()
		path := c.FullPath()
		if path == "" {
			path = "<unmatched>"
		}
		status := c.Writer.Status()
		if !c.Writer.Written() && status == http.StatusOK {
			status = 0
		}
		logger.Info("HTTP request",
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.Int("status", status),
			zap.Int("response_bytes", c.Writer.Size()),
			zap.Duration("duration", time.Since(started)),
		)
	}
}
