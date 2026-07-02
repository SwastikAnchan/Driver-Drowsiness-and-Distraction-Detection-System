class FatigueScorer:

    def calculate(
        self,
        ear,
        blink_count,
        yawn_count,
        distracted
    ):

        score = 0

        if ear < 0.22:
            score += 40

        if blink_count > 20:
            score += 20

        if yawn_count > 2:
            score += 30

        if distracted:
            score += 10

        return min(score,100)