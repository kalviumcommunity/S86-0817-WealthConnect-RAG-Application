from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time

class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        auth_header = self.headers.get('Authorization')
        
        # Test 401 Authentication Error
        if auth_header != 'Bearer valid-test-key':
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": {
                    "message": "Invalid API key provided.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }).encode())
            return
            
        # Optional: could test 429 by checking a header or path, but let's just do success for the successful output.
        
        # Success response (200)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        response_data = {
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I am a helpful RAG assistant designed to answer questions based on retrieved context."
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 18,
                "completion_tokens": 15,
                "total_tokens": 33
            },
            "system_fingerprint": "fp_mock"
        }
        self.wfile.write(json.dumps(response_data).encode())

def run(server_class=HTTPServer, handler_class=MockOpenAIHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting mock server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
