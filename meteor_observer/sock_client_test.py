import socket

ip = "127.0.0.1"
port = 8000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ip, port))

print("Connected!!!!!")

while True:
	print("<メッセージを入力してください>")
	message = input('>>>')
	if not message:
		s.send("quit".encode("utf-8"))
		break
	s.send(message.encode("utf-8"))
	
s.close()

print("END")