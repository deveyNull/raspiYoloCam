import io

import logging
import socketserver
import socket
from threading import Condition
from http import server
import struct
from PIL import Image

import cv2


import math

server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 7000))
server_socket.listen(0)

counter = 0


PAGE="""\
<html>
<head>
<title>RasPi Client-Server Demo</title>
</head>
<body>
<h1>RasPi Client-Server Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:

                while True:
                    # Read the length of the image as a 32-bit unsigned int. If the
                    # length is zero, quit the loop
                    image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                    if not image_len:
                        break
                    # Construct a stream to hold the image data and read the image
                    # data from the connection
                    image_stream = io.BytesIO()
                    image_stream.write(connection.read(image_len))
                    # Rewind the stream, open it as an image with PIL and do some
                    # processing on it
                    image_stream.seek(0)
                    frame = image_stream.read() #buf.read() #output.frame
                    """
                    

                    (H, W) = image.shape[:2]
                    # determine only the *output* layer names that we need from YOLO
                    ln = net.getLayerNames()
                    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                    # construct a blob from the input image and then perform a forward
                    # pass of the YOLO object detector, giving us our bounding boxes and
                    # associated probabilities
                    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                    net.setInput(blob)
                    start = gimmetime.time()
                    layerOutputs = net.forward(ln)

                    end = gimmetime.time()
                    # show timing information on YOLO
                    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))


                    # loop over each of the layer outputs
                    for output in layerOutputs:
                        # loop over each of the detections
                        for detection in output:
	                        # extract the class ID and confidence (i.e., probability) of
	                        # the current object detection
	                        scores = detection[5:]
	                        classID = np.argmax(scores)
	                        confidence = scores[classID]
	                        # filter out weak predictions by ensuring the detected
	                        # probability is greater than the minimum probability
	                        if confidence > .5: #magic number alert
		                        # scale the bounding box coordinates back relative to the
		                        # size of the image, keeping in mind that YOLO actually
		                        # returns the center (x, y)-coordinates of the bounding
		                        # box followed by the boxes' width and height
		                        box = detection[0:4] * np.array([W, H, W, H])
		                        (centerX, centerY, width, height) = box.astype("int")
		                        # use the center (x, y)-coordinates to derive the top and
		                        # and left corner of the bounding box
		                        x = int(centerX - (width / 2))
		                        y = int(centerY - (height / 2))
		                        # update our list of bounding box coordinates, confidences,
		                        # and class IDs
		                        boxes.append([x, y, int(width), int(height)])
		                        confidences.append(float(confidence))
		                        classIDs.append(classID)

                    # apply non-maxima suppression to suppress weak, overlapping bounding
                    # boxes
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, .5,
                    .3) #.3 = threshold, .5 = confidence

                    

                    # ensure at least one detection exists
                    if len(idxs) > 0:
                        # loop over the indexes we are keeping
                        for i in idxs.flatten():
                            # extract the bounding box coordinates
                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])
                            lx = x+(.5*w)
                            lxArray.append(lx)
	                        # draw a bounding box rectangle and label on the image
                            color = [int(c) for c in COLORS[classIDs[i]]]
                            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
		                        0.5, color, 2)
                    # show the output image
                    cv2.imshow("Image", image)
                    """
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                    
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
                
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


connection = server_socket.accept()[0].makefile('rb')
try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    print("Done")

