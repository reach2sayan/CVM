class Logger:

    def __init__(self, stream, logfile):
        self.stream = stream
        self.outfile = open(logfile,'w')
    
    def write(self, message):
        self.stream.write(message)
        self.outfile.write(message)
        self.outfile.flush()

    def flush(self, message):
        pass
