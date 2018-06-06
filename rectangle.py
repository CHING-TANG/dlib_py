

class rectangle(object):
    def __init__(self, l, t, r, b):
    self.l = l
    self.t = t
    self.r = r
    self.b = b

    def width ():
        if (is_empty()):
            return 0
        else:
            return self.r - self.l + 1
        

    def height(): 
        if (is_empty()):
            return 0
        else:
            return self.b - self.t + 1

    def area():
        return width()*height()

    def is_empty():
        return  self.t > self.b or self.l > self.r
