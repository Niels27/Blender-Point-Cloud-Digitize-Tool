#library imports
import bpy
from bpy import context
import os
import numpy as np
import asyncio
import websockets
import socket
import threading
import glob
import json

websocket_port = 8765 #port to use for websockets
active_websockets = set() #Global variable to store active websockets
websocket_server_manager=None #Global variable to store the websocket server manager

#Websocket handling
#Class to handle the websocket server 
class WebSocketServerManager:
    def __init__(self, port):
        self.port = port
        self.loop = None
        self.server = None
        self.thread = None

    def start_server(self):
        if self.thread is not None and self.thread.is_alive():
            print("WebSocket server is already running.")
            return

        #Set up a new event loop for the thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        #Start the server coroutine within this loop
        server_coro = websockets.serve(handle, "localhost", self.port)
        self.server = self.loop.run_until_complete(server_coro)

        #Run the loop in a separate thread
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        print("WebSocket server started on port", self.port)

    def stop_server(self):
        if self.loop is None or not self.loop.is_running():
            print("WebSocket server is not running.")
            return

        #Schedule the server to close and stop the loop
        self.loop.call_soon_threadsafe(self.server.close)
        self.loop.call_soon_threadsafe(self.loop.stop)

        #Wait for the thread to finish
        self.thread.join()
        self.thread = None
        self.loop = None
        print("WebSocket server stopped.")

    def restart_server(self):
        self.stop_server()
        self.start_server()
        
#update the websocket server when the checkbox is changed
def update_connect_socket(self, context):
    global websocket_server_manager,websocket_port
    if self.connect_socket:
        websocket_server_manager = WebSocketServerManager(port=websocket_port)
        websocket_server_manager.restart_server()
    else:
        if websocket_server_manager is not None:
            websocket_server_manager.stop_server()
        else:
            print("WebSocket server never started")
            
def run_asyncio_coroutine(coroutine):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coroutine)
        loop.close()
    except Exception as e:
        print("Error in run_asyncio_coroutine:", e)
          
#Function to execute operators at given coordinates
def execute_operators_at_coordinates(x,y,z,marker):
    
    #Create a Vector for the incoming coordinates 
    point = [x, y, z]
    if(marker=="triangle_marker"):
        bpy.ops.custom.mark_fixed_triangle(external_x=point[0], external_y=point[1], external_z=point[2])
    elif(marker=="complex_marker"):
        bpy.ops.custom.mark_complex(external_x=point[0], external_y=point[1], external_z=point[2])
    elif(marker=="None"):
        print("No marker selected")
    else:
        print("Unknown marker type")
        
    send_shape_to_webviewer=False
    if send_shape_to_webviewer:
        #Send message back to webviewer after execution
        #asyncio.run(send_message_to_websocket("new_shape_available"))
        threading.Thread(target=run_asyncio_coroutine, args=(send_message_to_websocket("new_shape_available"),)).start()
        
#Function to send data to the websocket
async def handle(websocket, path):
    global active_websockets
    active_websockets.add(websocket)
    print("WebSocket connected:", websocket)
    try:
        while True:
            message = await websocket.recv()
            print(f"Received data: {message}")
            
            #Parse the JSON data
            try:
                data = json.loads(message)  #Parse the JSON message
                if "message" in data:  #Check if 'message' key exists
                    msg_data = data["message"]  
                    if all(k in msg_data for k in ("x", "y", "z", "type")):
                        execute_operators_at_coordinates(msg_data["x"], msg_data["y"], msg_data["z"], msg_data["type"])
                    else:
                        print("Received data is not in the correct format")
            except json.JSONDecodeError:
                print("Invalid data format")  
    finally:
        active_websockets.remove(websocket)

#Function to send data to all connected websockets
async def send_message_to_websocket(message):
    try:
        if active_websockets:
            print("attempting to send message to websocket")
            await asyncio.wait([websocket.send(message) for websocket in active_websockets])
            print("Sent message: ", message, " to websockets")  
        else:
            print("No active websockets found")
    except Exception as e:
        print("Error sending message to websocket:", str(e))
       
#Function to send files over the websocket            
async def send_file_over_websocket(websocket, file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
        await websocket.send(data)