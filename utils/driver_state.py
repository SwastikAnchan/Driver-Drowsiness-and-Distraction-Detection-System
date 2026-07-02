class DriverState:

    def classify(self, score):

        if score >= 70:
            return "DROWSY"

        elif score >= 40:
            return "SLEEPY"

        return "ALERT"