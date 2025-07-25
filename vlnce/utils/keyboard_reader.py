from pynput import keyboard

class KeyboardReader:
    def __init__(self):
        self.key_pressed = None
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.key_pressed = key.char  # For regular keys
        except AttributeError:
            self.key_pressed = key.name  # For special keys

    def on_release(self, key):
        if key == keyboard.Key.esc:  # Stop listener on escape key
            return False
        self.key_pressed = None

    def get_key(self):
        return self.key_pressed
    
if __name__ == "__main__":
    reader = KeyboardReader()
    print("Press any key (ESC to exit):")
    try:
        while True:
            key = reader.get_key()
            if key is not None:
                print(f"Key pressed: {key}")
    except KeyboardInterrupt:
        print("Exiting...")