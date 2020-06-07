# this class defines the experimental flow, which methods need to be called in which order etc
class Experimentor:
    def __init__(self, writer):
        self.writer = writer
        self.
        print("i was born")

    def run_experiment(self):
        self.writer.write_scalar("boot", 10.0)
        self.writer.write_scalars({
            "boot1": 11.0,
            "boot2": 12.0,
            "boot3": 13.0,
        }
        )
        self.writer.save()
        print("gotcha")