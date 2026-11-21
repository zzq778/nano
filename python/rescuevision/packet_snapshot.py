"""Diagnostic packet persistence is optional and separate from physical UART writes."""
class PacketRecorder:
    def __init__(self,path,per_frame=False):self.path=path;self.per_frame=per_frame;self.packet=None
    def record(self,packet):
        self.packet=bytes(packet)
        if self.per_frame:self.path.write_bytes(self.packet)
    def close(self):
        if self.packet is not None and not self.per_frame:self.path.write_bytes(self.packet)
