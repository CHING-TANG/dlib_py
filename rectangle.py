import math

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
    
    def left(self):
        return self.l

    def top(self):
        return self.t

    def right(self):
        return self.r

    def bottom(self):
        return self.b

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



def move_rect (rect, p)   # const rectangle
# const rectangle& rect
# const point& p
    return rectangle(p.x(), p.y(), p.x()+rect.width()-1, p.y()+rect.height()-1);


def move_rect (rect, x, y)   # const rectangle
# const rectangle& rect
# long x,
# long y 
    return rectangle(x, y, x+rect.width()-1, y+rect.height()-1);


def set_rect_area(rect, area):  #  inline rectangle
# const rectangle& rect,
# unsigned long area  
    assert area > 0

    if (rect.area() == 0):
        # In this case we will make the output rectangle a square with the requested area.
        scale = round(math.sqrt(area))  # unsigned long 
        return centered_rect(rect, scale, scale)
    else:
        scale = math.sqrt(area / rect.area())  # double 
        return centered_rect(rect, round(rect.width()*scale), round(rect.height()*scale))

def centered_rect (x, y, width, height):  #  inline const rectangle
# long x,
# long y,
# unsigned long width,
# unsigned long height
    # rectangle result;
    # result.set_left ( x - static_cast<long>(width) / 2 );
    # result.set_top ( y - static_cast<long>(height) / 2 );
    # result.set_right ( result.left() + width - 1 );
    # result.set_bottom ( result.top() + height - 1 );
    l =  x - width / 2 
    t =  y - height / 2 
    r =  l + width - 1 
    b =  t + height - 1 
    return rectangle(l, t, r, b)


def centered_rect(rect, width, height): #  inline const rectangle
# const rectangle& rect,
# unsigned long width,
# unsigned long height
    return centered_rect((rect.left()+rect.right())/2,  (rect.top()+rect.bottom())/2, width, height)

def centered_rect(p, width, height):  #  inline const rectangle 
# const point& p,
# unsigned long width,
# unsigned long height
    return centered_rect(p.x(), p.y(), width, height)

