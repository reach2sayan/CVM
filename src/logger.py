class Logger:

    def __init__(self, stream, logfile, toscreen):
        self.toscreen = toscreen
        self.stream = stream
        self.outfile = open(logfile,'w')
    
    def write(self, message):
        if self.toscreen:
            self.stream.write(message)
        self.outfile.write(message)
        self.outfile.flush()

    def flush(self):
        pass
