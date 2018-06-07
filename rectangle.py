class rectangle(object):
    # WHAT THIS OBJECT REPRESENTS
    # This object represents a rectangular region inside a Cartesian 
    # coordinate system.  The region is the rectangle with its top 
    # left corner at position (left(),top()) and its bottom right corner 
    # at (right(),bottom()).

    # Note that the origin of the coordinate system, i.e. (0,0), is located
    # at the upper left corner.  That is, points such as (1,1) or (3,5) 
    # represent locations that are below and to the right of the origin.

    # Also note that rectangles where top() > bottom() or left() > right() 
    # represent empty rectangles.
    def __init__(self, l, t, r, b):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
    # r > l && b > t
    
    def width(self):
        if (self.is_empty()):
            return 0
        else:
            return self.r - self.l + 1
        

    def height(self): 
        if (self.is_empty()):
            return 0
        else:
            return self.b - self.t + 1

    def area(self):
        return self.width() * self.height()

    def is_empty(self):
        return self.t > self.b or self.l > self.r
    
    def intersect (self, rhs):
    # const rectangle& rhs
        return rectangle (
            max(self.l, rhs.l),
            max(self.t, rhs.t),
            min(self.r, rhs.r),
            min(self.b, rhs.b)
            )


    def operator_plus(self, rhs):
        if (rhs.is_empty()):
            return self
        elif(self.is_empty()):
            return rhs
        
        return rectangle (
            min(self.l, rhs.l),
            min(self.t, rhs.t),
            max(self.r, rhs.r),
            max(self.b, rhs.b)
            )