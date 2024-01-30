
import torch
from torchvision import transforms
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import base64
import time
from io import BytesIO
from PIL import Image

# Load your ResNet18 model
from ResNet18 import ResNet18

# Check if CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and set it to evaluation mode
model = ResNet18(dropout_rate=0).to(device)
model.load_state_dict(torch.load('model.pkl', map_location=device))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])


def base64_to_image(base64_string):
    try:
        # Split the data URI to extract the base64-encoded image data
        parts = base64_string.split(',')
        if len(parts) != 2:
            raise ValueError("Invalid data URI format")

        # Decode the base64 data
        image_data = base64.b64decode(parts[1])

        # Create an image object
        image = Image.open(BytesIO(image_data))

        return image
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None


# Define CORS headers
def set_cors_headers(self):
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
    self.send_header('Access-Control-Allow-Headers', 'Content-Type')


class GenderDetector(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        # Handle preflight CORS requests
        self.send_response(200)
        set_cors_headers(self)
        self.end_headers()

    def do_POST(self):
        if self.path == '/predict/gender':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            post_data = json.loads(post_data)

            response = {}
            if 'picture' in post_data:
                try:
                    image = base64_to_image(post_data['picture'])
                    image = image.convert('RGB')

                    image = transform(image).unsqueeze(0).to(device)  # Add a batch dimension and move to the same device as the model

                    # Make a prediction using the model
                    with torch.no_grad():
                        output = model(image)
                        output2 = model(image)
                        output3 = model(image)
                        output4 = model(image)
                        time.sleep(1)  # Just slows down the response, but it gives a cool effect :)

                    # Assuming your model outputs a single value between 0 and 1 (e.g., a probability)
                    predicted_value = (output.item() + output2.item() + output3.item() + output4.item())/4

                    self.send_response(200)
                    set_cors_headers(self)
                    self.end_headers()
                    self.wfile.write(str(predicted_value).encode('utf-8'))

                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    set_cors_headers(self)
                    self.end_headers()
                    self.wfile.write(b'Internal Server Error')

            else:
                self.send_response(400)
                set_cors_headers(self)
                self.end_headers()
                self.wfile.write(b'Could not process your picture.')

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

if __name__ == "__main__":
    serverPort = 8082
    hostName = "localhost"
    webServer = HTTPServer((hostName, serverPort), GenderDetector)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")