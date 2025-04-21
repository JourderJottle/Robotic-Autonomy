from movement import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.send_movement(0.3, 0)
    controller.send_movement(0.0, 0.0)
