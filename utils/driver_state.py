# driver_state.py

class DriverStateClassifier:

    def __init__(self):

        self.state = "Awake"

    # =====================================================
    # RULE BASED ML CLASSIFIER
    # =====================================================

    

        def classify(
            self,
            fatigue_score,
            eye_closure_duration,
            blink_rate,
            yawn_frequency,
            head_down,
            distracted
        ):

            # =====================================================
            # DISTRACTED
            # =====================================================

            if distracted:

                self.state = "Distracted"

            # =====================================================
            # VERY DROWSY
            # =====================================================

            elif (
                fatigue_score > 70 or
                eye_closure_duration > 2.5 or
                yawn_frequency >= 4
            ):

                self.state = "Very Drowsy"

            # =====================================================
            # SLEEPY
            # =====================================================

            elif (
                fatigue_score > 55 or
                blink_rate > 25
            ):

                self.state = "Sleepy"

            # =====================================================
            # HEAD DOWN
            # =====================================================

            elif head_down:

                self.state = "Distracted"

            # =====================================================
            # AWAKE
            # =====================================================

            else:

                self.state = "Awake"

            return self.state