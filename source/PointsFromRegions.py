# TagLab
# A semi-automatic segmentation tool
#
# Point annotations from the regions, e.g. as training data for CoralNet (Points > Label Points from Regions):
# the points of the map get the class of the region under them, and points are added inside the regions (one in
# every region without a point, then more until every class has a minimum number of points). A point closer than
# the edge distance to another class, or to an area without regions, is left Empty: its label is uncertain.
# Nothing here changes the annotations: the caller applies the results.

import math
import numpy as np
import cv2
from collections import Counter

EMPTY = "Empty"
DEFAULT_EDGE_MM = 2.0       # points closer than this to another class are left Empty
DEFAULT_PER_CLASS = 20      # minimum number of points of each class
MIN_SPACING_PX = 5          # minimum distance (pixels) of the added points from the other points
ATTEMPTS_PER_POINT = 200    # random positions tried for each point added inside a region
SEED = 1                    # the same options give the same points


def edgePixels(edge_mm, pixel_size_mm):
    """
    The edge distance in pixels of the map (rounded up).
    """
    if edge_mm <= 0.0 or pixel_size_mm <= 0.0:
        return 0
    return int(math.ceil(edge_mm / pixel_size_mm - 1e-9))


def insideArea(x, y, area):
    """
    True if the pixel is inside area = [top, left, width, height] (None: no limit).
    """
    if area is None:
        return True
    return area[1] <= x < area[1] + area[2] and area[0] <= y < area[0] + area[3]


class RegionLookup(object):
    """
    The class of the map at a pixel, from the regions (Empty regions are ignored). Where regions overlap, the
    smallest one wins (e.g. a sponge drawn on a sand region without a hole for it). If outside_class is given, the
    pixels without regions are of that class. The masks of the regions are computed when first needed.
    map_size: (width, height) of the map. The pixels beyond the map border are not known, so they never make an
    edge (without map_size they count as pixels without regions).
    """

    def __init__(self, blobs, edge_px, outside_class=None, map_size=None):

        # larger regions first: painted in this order, the smallest region wins
        self.blobs = sorted([b for b in blobs if b.class_name != EMPTY], key=lambda b: -float(b.area))
        self.r = max(int(edge_px), 0)
        self.outside_class = outside_class
        self.map_size = map_size

        names = sorted(set(b.class_name for b in self.blobs) | ({outside_class} if outside_class else set()))
        self.classes = [None] + names                       # class index -> name (0: no class)
        index = {name: i for i, name in enumerate(self.classes)}
        self.outside_index = index[outside_class] if outside_class else 0
        # class index of each region, then of "no region" (region index -1 -> last element)
        self.region_class = np.array([index[b.class_name] for b in self.blobs] + [self.outside_index], dtype=np.int32)

        boxes = np.array([[int(v) for v in b.bbox] for b in self.blobs], dtype=np.int64).reshape(-1, 4)
        self.top = boxes[:, 0]
        self.left = boxes[:, 1]
        self.bottom = boxes[:, 0] + boxes[:, 3]
        self.right = boxes[:, 1] + boxes[:, 2]
        self.masks = {}

        yy, xx = np.mgrid[-self.r:self.r + 1, -self.r:self.r + 1]
        self.disc = (xx * xx + yy * yy) <= self.r * self.r

    def mask(self, i):
        m = self.masks.get(i)
        if m is None:
            m = self.blobs[i].getMask() > 0
            self.masks[i] = m
        return m

    def regionWindow(self, x, y, r):
        """
        Region index (-1: none) of the pixels of the (2r+1) x (2r+1) window centred on (x, y).
        """
        size = 2 * r + 1
        x0 = x - r
        y0 = y - r
        win = np.full((size, size), -1, dtype=np.int64)
        if len(self.blobs) == 0:
            return win

        selected = np.nonzero((self.left < x0 + size) & (self.right > x0) &
                              (self.top < y0 + size) & (self.bottom > y0))[0]
        for i in selected:
            t = int(self.top[i])
            l = int(self.left[i])
            r0 = max(y0, t)
            r1 = min(y0 + size, int(self.bottom[i]))
            c0 = max(x0, l)
            c1 = min(x0 + size, int(self.right[i]))
            sub = self.mask(i)[r0 - t:r1 - t, c0 - l:c1 - l]
            win[r0 - y0:r1 - y0, c0 - x0:c1 - x0][sub] = i
        return win

    def at(self, x, y):
        """
        (class name or None, near an edge, region index or -1) at the pixel (x, y). Near an edge: a pixel closer
        than the edge distance has another class (or none).
        """
        x = int(round(x))
        y = int(round(y))
        r = self.r
        win = self.regionWindow(x, y, r)
        classes = self.region_class[win]
        c = classes[r, r]
        near = False
        if r > 0:
            neighbours = self.disc
            if self.map_size is not None:
                # only the pixels of the map
                neighbours = neighbours.copy()
                neighbours[:max(r - y, 0), :] = False
                neighbours[:, :max(r - x, 0)] = False
                neighbours[max(self.map_size[1] - y + r, 0):, :] = False
                neighbours[:, max(self.map_size[0] - x + r, 0):] = False
            near = bool(np.any(classes[neighbours] != c))
        return self.classes[c], near, int(win[r, r])


def labelPoints(points, lookup, only_empty=True):
    """
    The class each point gets from the region under it (the class of the lookup's outside_class where there is no
    region): Empty if it is near an edge or on no region. Only the Empty points are considered if only_empty.
    Returns (changes [(point, new class)] of the points whose class changes, final {point: class} of the points
    considered, counts: Counter of the final classes, plus 'near edge' and 'no region' counts).
    """
    changes = []
    final = {}
    counts = Counter()
    for point in points:
        if only_empty and point.class_name != EMPTY:
            continue
        name, near, region = lookup.at(point.coordx, point.coordy)
        if name is None:
            new_class = EMPTY
            counts["no region"] += 1
        elif near:
            new_class = EMPTY
            counts["near edge"] += 1
        else:
            new_class = name
            counts[name] += 1
        final[point] = new_class
        if new_class != point.class_name:
            changes.append((point, new_class))
    return changes, final, counts


class SpacingGrid(object):
    """
    Points kept at least `spacing` pixels apart (grid of cells of that size).
    """

    def __init__(self, spacing):
        self.spacing = max(float(spacing), 1.0)
        self.cells = {}

    def add(self, x, y):
        key = (int(x // self.spacing), int(y // self.spacing))
        self.cells.setdefault(key, []).append((x, y))

    def free(self, x, y):
        cx = int(x // self.spacing)
        cy = int(y // self.spacing)
        s2 = self.spacing * self.spacing
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for (px, py) in self.cells.get((i, j), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < s2:
                        return False
        return True


def addRegionPoints(lookup, labelled, per_class_min, area=None, seed=SEED):
    """
    Points to add inside the regions: one in every region without a point, then more until every class has at
    least per_class_min points, in regions chosen in proportion to their area. A new point is where the region is
    on top (not under a smaller region), not near an edge, and at least max(2 * edge, MIN_SPACING_PX) pixels from
    the other points; a region without such a pixel gets its point at its most interior pixel.
    labelled: [(x, y, class)] of the points of the map: all keep the spacing, those with a class count.
    area: [top, left, width, height] (e.g. the working area) or None; regions and points outside it are ignored.
    Returns [(x, y, class)].
    """
    rng = np.random.default_rng(seed)
    grid = SpacingGrid(max(2 * lookup.r, MIN_SPACING_PX))

    # regions (partly) inside the area, with their box clipped to it
    regions = []
    for i, blob in enumerate(lookup.blobs):
        t, l, b, r = int(lookup.top[i]), int(lookup.left[i]), int(lookup.bottom[i]), int(lookup.right[i])
        if area is not None:
            t, l = max(t, int(area[0])), max(l, int(area[1]))
            b, r = min(b, int(area[0] + area[3])), min(r, int(area[1] + area[2]))
        if b > t and r > l:
            regions.append((i, t, l, b, r))

    # the existing points: per region and per class
    per_region = Counter()
    per_class = Counter()
    for (x, y, name) in labelled:
        if not insideArea(x, y, area):
            continue
        grid.add(x, y)
        per_class[name] += 1
        _, _, region = lookup.at(x, y)
        if region >= 0 and lookup.blobs[region].class_name == name:
            per_region[region] += 1

    added = []

    def sample(i, t, l, b, r):
        m = lookup.mask(i)
        top, left = int(lookup.top[i]), int(lookup.left[i])
        for _ in range(ATTEMPTS_PER_POINT):
            x = int(rng.integers(l, r))
            y = int(rng.integers(t, b))
            if not m[y - top, x - left]:
                continue
            name, near, region = lookup.at(x, y)
            if near or region != i or not grid.free(x, y):
                continue
            return x, y
        return None

    def mostInterior(i, t, l, b, r):
        # the pixel of the region (where it is on top) farthest from its border
        top, left = int(lookup.top[i]), int(lookup.left[i])
        own = lookup.mask(i)[t - top:b - top, l - left:r - left].copy()
        # the regions after i in the lookup (smaller) are on top of it
        for j in np.nonzero((lookup.left < r) & (lookup.right > l) & (lookup.top < b) & (lookup.bottom > t))[0]:
            if j > i:
                tj, lj = int(lookup.top[j]), int(lookup.left[j])
                r0, r1 = max(t, tj), min(b, int(lookup.bottom[j]))
                c0, c1 = max(l, lj), min(r, int(lookup.right[j]))
                if r1 > r0 and c1 > c0:
                    sub = lookup.mask(j)[r0 - tj:r1 - tj, c0 - lj:c1 - lj]
                    own[r0 - t:r1 - t, c0 - l:c1 - l][sub] = False
        if not own.any():
            return None
        padded = np.pad(own.astype(np.uint8), 1)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return int(l + x), int(t + y)

    # one point in every region without a point (smallest first)
    for (i, t, l, b, r) in sorted(regions, key=lambda reg: float(lookup.blobs[reg[0]].area)):
        if per_region[i] > 0:
            continue
        pos = sample(i, t, l, b, r)
        if pos is None:
            pos = mostInterior(i, t, l, b, r)
        if pos is None:
            continue
        name = lookup.blobs[i].class_name
        added.append((pos[0], pos[1], name))
        grid.add(*pos)
        per_region[i] += 1
        per_class[name] += 1

    # more points until every class has per_class_min, in regions chosen in proportion to their area
    by_class = {}
    for reg in regions:
        by_class.setdefault(lookup.blobs[reg[0]].class_name, []).append(reg)
    for name in sorted(by_class):
        missing = per_class_min - per_class[name]
        if missing <= 0:
            continue
        candidates = by_class[name]
        weights = np.array([float(lookup.blobs[reg[0]].area) for reg in candidates])
        weights = weights / weights.sum() if weights.sum() > 0 else None
        failures = 0
        while missing > 0 and failures < 20 + 5 * per_class_min:
            reg = candidates[int(rng.choice(len(candidates), p=weights))]
            pos = sample(*reg)
            if pos is None:
                failures += 1
                continue
            added.append((pos[0], pos[1], name))
            grid.add(*pos)
            per_class[name] += 1
            missing -= 1

    return added
