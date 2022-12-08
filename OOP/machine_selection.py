from abc import abstractmethod


class Machine():
    self.name : str
    self.channel : int
    self.type : str
    
    def set_channel(self, channel_selected : int):
        self.channel = channel_selected

class fan(Machine):
    name = 'fan'
    channel = 0
    type = 'stationnary'

class vanne(Machine):
    name ='vanne'
    channel = 2
    type = 'non-stationnary'

class motor(Machine):
    name = 'motor'
    channel = 4
    type = 'stationnary'

class pump(Machine):
    name = 'pump'
    channel = 6
    type = 'non-stationnary'


