class PID:
    def __init__(self, Kp=0, Ki=0, Kd=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
    
    def compute(self, error, derivative):
        return self.Kp * error + self.Kd * derivative
        