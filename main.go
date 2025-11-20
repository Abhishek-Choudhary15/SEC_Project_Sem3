package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	maxUploadSize   = 8 << 20           // 8 MB max upload
	uploadPath      = "./uploads"       // where images are stored
	modelScriptPath = "./model_infer.py" // Python script path
	serveAddr       = ":8080"
)

// ModelResult is the structure we expect from the Python script.
type ModelResult struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
	// Add more fields if your model returns them (e.g. "score", "explanation")
}

// apiError is used for error responses in JSON.
type apiError struct {
	Error string `json:"error"`
}

func main() {
	// Ensure upload directory exists
	if err := os.MkdirAll(uploadPath, 0755); err != nil {
		log.Fatalf("failed to create upload dir: %v", err)
	}

	mux := http.NewServeMux()

	// Serve frontend static files from ./frontend
	fs := http.FileServer(http.Dir("./frontend"))
	mux.Handle("/", fs)

	// Upload API
	mux.HandleFunc("/upload", uploadHandler)

	srv := &http.Server{
		Addr:         serveAddr,
		Handler:      loggingMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("Server listening on http://localhost%s", serveAddr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}

// loggingMiddleware logs each request.
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

// writeJSON writes JSON with given status code.
func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// uploadHandler handles POST /upload
func uploadHandler(w http.ResponseWriter, r *http.Request) {
	// CORS preflight (if you need cross-origin)
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, apiError{Error: "method not allowed"})
		return
	}

	// Limit request body size
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize+512)
	if err := r.ParseMultipartForm(maxUploadSize); err != nil {
		writeJSON(w, http.StatusBadRequest, apiError{Error: "file too large or invalid form data"})
		return
	}

	file, hdr, err := r.FormFile("image")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, apiError{Error: "missing 'image' field"})
		return
	}
	defer file.Close()

	// Read head bytes to detect content type
	head := make([]byte, 512)
	n, _ := file.Read(head)
	contentType := http.DetectContentType(head[:n])
	if !strings.HasPrefix(contentType, "image/") {
		writeJSON(w, http.StatusBadRequest, apiError{Error: "uploaded file is not an image"})
		return
	}

	// Reset file reader
	if seeker, ok := file.(io.Seeker); ok {
		_, _ = seeker.Seek(0, io.SeekStart)
	}

	// Determine file extension
	exts, _ := mime.ExtensionsByType(contentType)
	ext := ""
	if len(exts) > 0 {
		ext = exts[0]
	} else {
		ext = filepath.Ext(hdr.Filename)
	}
	filename := fmt.Sprintf("img_%d%s", time.Now().UnixNano(), ext)
	dstPath := filepath.Join(uploadPath, filename)

	// Save file
	out, err := os.Create(dstPath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, apiError{Error: "unable to save file"})
		return
	}
	defer out.Close()

	if _, err := io.Copy(out, file); err != nil {
		writeJSON(w, http.StatusInternalServerError, apiError{Error: "unable to write file"})
		return
	}

	// Run model inference via Python script
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	res, err := runModelScript(ctx, modelScriptPath, dstPath)
	if err != nil {
		log.Printf("model error: %v", err)
		writeJSON(w, http.StatusInternalServerError, apiError{Error: "model inference failed: " + err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, res)
}

// runModelScript runs `python3 model_infer.py <imagePath>` and parses JSON.
func runModelScript(ctx context.Context, scriptPath, imagePath string) (*ModelResult, error) {
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, errors.New("model script not found")
	}

	// Use "python" instead of "python3" if that's how your system is set up
	cmd := exec.CommandContext(ctx, "python3", scriptPath, imagePath)
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return nil, errors.New("model timed out")
	}
	if err != nil {
		return nil, fmt.Errorf("script error: %v output: %s", err, string(out))
	}

	var mr ModelResult
	if err := json.Unmarshal(out, &mr); err != nil {
		return nil, fmt.Errorf("invalid JSON from model: %v output: %s", err, string(out))
	}

	// Basic sanity checks
	if mr.Label == "" {
		return nil, fmt.Errorf("model returned empty label")
	}
	if mr.Confidence < 0 || mr.Confidence > 1 {
		log.Printf("warning: confidence out of bounds: %v", mr.Confidence)
	}
	return &mr, nil
}
