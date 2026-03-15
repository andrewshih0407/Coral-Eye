import asyncio
import serial
import serial.tools.list_ports
import websockets

COM_PORT  = "COM3"
BAUD_RATE = 9600
WS_PORT   = 8765

clients = set()
ser     = None

def find_arduino():
    ports = serial.tools.list_ports.comports()
    print("\nAvailable COM ports:")
    for p in ports:
        print(f"  {p.device} — {p.description}")
        if "CH340" in p.description or "Arduino" in p.description:
            return p.device
    return COM_PORT

async def ws_handler(websocket):
    clients.add(websocket)
    print("Website connected")
    try:
        async for msg in websocket:
            pass
    except:
        pass
    finally:
        clients.discard(websocket)
        print("Website disconnected")

async def read_serial():
    global ser
    port = find_arduino()
    print(f"\nConnecting to Arduino on {port}...")
    while True:
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
            print(f"Connected — reading sensor data...\n")
            buf = ""
            while True:
                await asyncio.sleep(0.05)
                if ser.in_waiting:
                    raw = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
                    buf += raw
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("{"):
                            print(f"<- {line}")
                            for ws in list(clients):
                                try:
                                    await ws.send(line)
                                except:
                                    clients.discard(ws)
        except serial.SerialException as e:
            print(f"Serial error: {e} — retrying in 3s...")
            await asyncio.sleep(3)
        finally:
            if ser and ser.is_open:
                ser.close()

async def main():
    print("PaleWatch USB Bridge")
    print("====================")
    async with websockets.serve(ws_handler, "localhost", WS_PORT):
        await read_serial()

asyncio.run(main())
