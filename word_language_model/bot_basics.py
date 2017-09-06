import threading

# class InputThread(threading.Thread):
# 	def run(self):
# 		self.daemon = True
# 		while True:
# 			self.last_user_input = input('type something:')
# 			print(len(self.last_user_input))


# it = InputThread()
# it.start()


while True:
	it = input("YOU:")
	print("BRERIN: ",len(it))
