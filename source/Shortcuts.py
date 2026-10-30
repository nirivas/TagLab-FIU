# TagLab
# A semi-automatic segmentation tool
#
# Keyboard and mouse shortcuts (File > Keyboard and Mouse Shortcuts...): the actions that have a key, their default
# keys, the keys and mouse buttons chosen by the user and the matching of key and mouse events. The choices are saved
# in TagLab's settings, group "lab-shortcuts" (the official TagLab, which shares the settings, ignores it); only the
# values that differ from the defaults are stored.
#
# Kinds of actions:
#   QACTION  a menu action: its key is the QAction shortcut (TagLab.applyShortcuts)
#   COMMAND  run by TagLab.keyPressEvent / TagLab.runShortcut
#   FIXED    a command whose key can't be changed (Space, Esc: every tool's instructions name them); it can get a mouse
#            button
#   REVIEW   the carry-over review panel (QtCarryOverReview); while it is open its keys come first
#   MOVE     the Move Region tool keys (QtImageViewerPlus.moveRegionKey): plain keys, SHIFT gives the bigger step

import logging

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QKeySequence

logfile = logging.getLogger("tool-logger")

QACTION = "qaction"
COMMAND = "command"
FIXED = "fixed"
REVIEW = "review"
MOVE = "move"

GLOBAL_KINDS = [QACTION, COMMAND, FIXED]

GROUPS = ["File and edit", "Regions", "View", "Tools", "Confirm and cancel", "Carry-over review", "Move Region tool"]

ACTIONS = [
    # id, group, label, default key, kind
    ("new_project", "File and edit", "New project", "Ctrl+N", QACTION),
    ("open_project", "File and edit", "Open project", "Ctrl+O", QACTION),
    ("save_project", "File and edit", "Save project", "Ctrl+S", QACTION),
    ("save_as", "File and edit", "Save project as", "Ctrl+Alt+S", QACTION),
    ("add_map", "File and edit", "Add new map", "Ctrl+L", QACTION),
    ("maps_editor", "File and edit", "Maps editor", "Ctrl+M", QACTION),
    ("help", "File and edit", "Help (list of the main keys)", "Ctrl+H", QACTION),
    ("undo", "File and edit", "Undo", "Ctrl+Z", QACTION),
    ("redo", "File and edit", "Redo", "Ctrl+Shift+Z", QACTION),
    ("select_all", "File and edit", "Select all", "Ctrl+A", QACTION),
    ("select_none", "File and edit", "Select none", "Ctrl+D", QACTION),
    ("select_invert", "File and edit", "Invert selection", "Ctrl+I", QACTION),

    ("assign", "Regions", "Assign class to region", "A", QACTION),
    ("delete", "Regions", "Delete region", "Del", QACTION),
    ("merge", "Regions", "Merge overlapping regions", "M", QACTION),
    ("divide", "Regions", "Divide regions", "D", QACTION),
    ("subtract", "Regions", "Subtract regions", "S", QACTION),
    ("snap", "Regions", "Snap borders", "B", QACTION),
    ("fill", "Regions", "Fill region", "F", QACTION),
    ("refine", "Regions", "Refine border", "R", QACTION),
    ("dilate", "Regions", "Dilate border", "+", QACTION),
    ("erode", "Regions", "Erode border", "-", QACTION),
    ("background_region", "Regions", "Create a background region using the working area", "N", QACTION),

    ("split_screen", "View", "Split screen on / off", "Alt+S", QACTION),
    ("switch_channel", "View", "Switch RGB / DEM", "C", COMMAND),
    ("toggle_fill", "View", "Show / hide the fill of the regions", "Q", COMMAND),
    ("toggle_borders", "View", "Show / hide the borders", "W", COMMAND),
    ("toggle_ids", "View", "Show / hide the region ids", "E", COMMAND),
    ("toggle_grid", "View", "Show / hide the grid", "G", COMMAND),
    ("grid_cell", "View", "Grid: change the state of the cell under the mouse", "U", COMMAND),

    ("tool_pan", "Tools", "Pan / zoom", "1", COMMAND),
    ("tool_fourclicks", "Tools", "4-clicks segmentation", "2", COMMAND),
    ("tool_ritm", "Tools", "Positive / negative clicks segmentation (RITM)", "3", COMMAND),
    ("tool_freehand", "Tools", "Freehand segmentation", "4", COMMAND),
    ("tool_assign", "Tools", "Assign class (tool)", "5", COMMAND),
    ("tool_editborder", "Tools", "Edit region border", "6", COMMAND),
    ("tool_cut", "Tools", "Cut region", "7", COMMAND),
    ("tool_crack", "Tools", "Create crack", "8", COMMAND),
    ("tool_ruler", "Tools", "Measure tool (ruler)", "9", COMMAND),
    ("classifier", "Tools", "Fully automatic segmentation (classifier)", "0", COMMAND),
    ("tool_moveregion", "Tools", "Move region", "", COMMAND),
    ("tool_point", "Tools", "Place annotation point", "", COMMAND),
    ("tool_watershed", "Tools", "Watershed segmentation", "", COMMAND),
    ("tool_saminteractive", "Tools", "SAM: positive / negative clicks in an area", "", COMMAND),
    ("tool_sam", "Tools", "SAM: all instances in an area", "", COMMAND),
    ("tool_match", "Tools", "Edit matches", "", COMMAND),

    ("confirm", "Confirm and cancel", "Confirm the operation of the tool", "Space", FIXED),
    ("cancel", "Confirm and cancel", "Cancel the operation of the tool", "Esc", FIXED),

    ("review_next", "Carry-over review", "Next colony", "N", REVIEW),
    ("review_previous", "Carry-over review", "Previous colony", "P", REVIEW),
    ("review_keep", "Carry-over review", "Keep the colony", "K", REVIEW),
    ("review_remove", "Carry-over review", "Remove the colony", "Del", REVIEW),

    ("move_left", "Move Region tool", "Move left (SHIFT: 10 pixels)", "Left", MOVE),
    ("move_right", "Move Region tool", "Move right (SHIFT: 10 pixels)", "Right", MOVE),
    ("move_up", "Move Region tool", "Move up (SHIFT: 10 pixels)", "Up", MOVE),
    ("move_down", "Move Region tool", "Move down (SHIFT: 10 pixels)", "Down", MOVE),
    ("turn_left", "Move Region tool", "Turn 5 degrees counter-clockwise (SHIFT: 90 degrees)", ",", MOVE),
    ("turn_right", "Move Region tool", "Turn 5 degrees clockwise (SHIFT: 90 degrees)", ".", MOVE),
    ("turn_left_90", "Move Region tool", "Turn 90 degrees counter-clockwise", "<", MOVE),
    ("turn_right_90", "Move Region tool", "Turn 90 degrees clockwise", ">", MOVE),
]

IDS = [a[0] for a in ACTIONS]
GROUP = {a[0]: a[1] for a in ACTIONS}
LABEL = {a[0]: a[2] for a in ACTIONS}
DEFAULT_KEY = {a[0]: a[3] for a in ACTIONS}
KIND = {a[0]: a[4] for a in ACTIONS}

# Move Region steps: ("nudge", dx, dy) in pixels or ("turn", degrees clockwise on the screen)
MOVE_STEPS = {"move_left": ("nudge", -1, 0), "move_right": ("nudge", 1, 0), "move_up": ("nudge", 0, -1),
              "move_down": ("nudge", 0, 1), "turn_left": ("turn", -5), "turn_right": ("turn", 5),
              "turn_left_90": ("turn", -90), "turn_right_90": ("turn", 90)}

# keys that can't be given to an action: Space and Esc (confirm / cancel), Tab (ends the editing of a border) and the
# upstream border-refinement keys Home, End, [ and ] (TagLab.keyPressEvent)
RESERVED_KEYS = ["Space", "Esc", "Tab", "Home", "End", "[", "]"]

# mouse buttons that can be given to an action (pressed without SHIFT / CTRL / ALT)
MOUSE_BUTTONS = [("Middle", Qt.MiddleButton, "middle button"),
                 ("Back", Qt.BackButton, "thumb button (back)"),
                 ("Forward", Qt.ForwardButton, "thumb button (forward)"),
                 ("Right", Qt.RightButton, "right button")]
MOUSE_NAMES = [m[0] for m in MOUSE_BUTTONS]
MOUSE_TEXT = {m[0]: m[2] for m in MOUSE_BUTTONS}

MODIFIERS = int(Qt.ControlModifier | Qt.AltModifier | Qt.ShiftModifier | Qt.MetaModifier)

SETTINGS_GROUP = "lab-shortcuts"


def defaultSettings():
    return QSettings("VCLAB", "TagLab")


# where the choices are saved (tests replace it with an INI file)
settingsFactory = defaultSettings


def keyCode(text):
    """
    The key + modifiers of a key text ("Ctrl+S", "K", "Del"), 0 for no key or a text that is not a single key.
    """
    if not text:
        return 0
    seq = QKeySequence(text, QKeySequence.PortableText)
    if seq.count() != 1 or seq[0] == Qt.Key_unknown:
        return 0
    return int(seq[0])


def keyText(code, native=True):
    if code == 0:
        return ""
    return QKeySequence(code).toString(QKeySequence.NativeText if native else QKeySequence.PortableText)


def eventCode(key, modifiers):
    return int(key) | (int(modifiers) & MODIFIERS)


def isLetter(key):
    return int(Qt.Key_A) <= int(key) <= int(Qt.Key_Z)


def plainCode(code):
    return code & ~MODIFIERS


def mouseName(button):
    for name, b, text in MOUSE_BUTTONS:
        if int(b) == int(button):
            return name
    return None


class Bindings(object):
    """
    The key (as an int: key + modifiers, 0 = none) and the mouse button (a name of MOUSE_NAMES or "") of every action.
    """

    def __init__(self):
        self.keys = {}
        self.mouse = {}
        self.resetDefaults()

    def resetDefaults(self):
        self.keys = {name: keyCode(DEFAULT_KEY[name]) for name in IDS}
        self.mouse = {name: "" for name in IDS}

    def copy(self):
        other = Bindings()
        other.keys = dict(self.keys)
        other.mouse = dict(self.mouse)
        return other

    def setFrom(self, other):
        self.keys = dict(other.keys)
        self.mouse = dict(other.mouse)

    def setKey(self, name, code):
        if KIND[name] == FIXED:
            return
        if KIND[name] == MOVE:
            code = plainCode(code)
        self.keys[name] = int(code)

    def setMouse(self, name, mouse):
        self.mouse[name] = mouse if mouse in MOUSE_NAMES else ""

    # ---- settings ------------------------------------------------------------------------------------------------------

    def load(self):
        """
        Read the keys and buttons saved in the settings; a missing or unreadable value keeps the default.
        """
        self.resetDefaults()
        try:
            settings = settingsFactory()
            settings.beginGroup(SETTINGS_GROUP)
            for name in IDS:
                text = settings.value(name + "/key", None)
                if text is not None and KIND[name] != FIXED:
                    text = str(text)
                    code = keyCode(text)
                    if text == "" or code != 0:
                        self.setKey(name, code)
                    else:
                        logfile.warning("[SHORTCUTS] Key '" + text + "' of '" + name + "' not understood: default kept.")
                mouse = settings.value(name + "/mouse", None)
                if mouse is not None:
                    mouse = str(mouse)
                    if mouse == "" or mouse in MOUSE_NAMES:
                        self.mouse[name] = mouse
                    else:
                        logfile.warning("[SHORTCUTS] Mouse button '" + mouse + "' of '" + name + "' not understood.")
            settings.endGroup()
        except Exception:
            logfile.exception("[SHORTCUTS] Could not read the shortcuts: defaults used.")
            self.resetDefaults()

    def save(self):
        """
        Write the values that differ from the defaults (the group is rewritten).
        """
        settings = settingsFactory()
        settings.beginGroup(SETTINGS_GROUP)
        settings.remove("")
        for name in IDS:
            if self.keys[name] != keyCode(DEFAULT_KEY[name]):
                settings.setValue(name + "/key", keyText(self.keys[name], native=False))
            if self.mouse[name] != "":
                settings.setValue(name + "/mouse", self.mouse[name])
        settings.endGroup()
        settings.sync()

    def isDefault(self):
        return all(self.keys[name] == keyCode(DEFAULT_KEY[name]) and self.mouse[name] == "" for name in IDS)

    # ---- matching --------------------------------------------------------------------------------------------------------

    def keySequence(self, name):
        code = self.keys.get(name, 0)
        return QKeySequence(code) if code != 0 else QKeySequence()

    def actionForKey(self, key, modifiers, kinds):
        """
        The action of the given kinds with this key, None if none. When nothing matches and SHIFT is held on a key that is
        not a letter, the key without SHIFT is tried as well (as Qt does for the menu shortcuts: "+" or "<" are typed with
        SHIFT on many keyboards).
        """
        if int(key) in (0, int(Qt.Key_unknown), int(Qt.Key_Shift), int(Qt.Key_Control), int(Qt.Key_Alt),
                        int(Qt.Key_Meta)):
            return None
        code = eventCode(key, modifiers)
        codes = [code]
        if code & int(Qt.ShiftModifier) and not isLetter(key):
            codes.append(code & ~int(Qt.ShiftModifier))
        for c in codes:
            for name in IDS:
                if KIND[name] in kinds and self.keys[name] == c:
                    return name
        return None

    def moveForKey(self, key, modifiers):
        """
        The Move Region step of a key: ("nudge", dx, dy) or ("turn", degrees), None if it is not a Move Region key. Only the
        key counts (CTRL and ALT are ignored, as before); SHIFT gives 10 pixels, or 90 degrees with the 5-degree keys.
        """
        key = int(key)
        shift = bool(int(modifiers) & int(Qt.ShiftModifier))
        for name in ("turn_left_90", "turn_right_90", "move_left", "move_right", "move_up", "move_down", "turn_left",
                     "turn_right"):
            if self.keys[name] != 0 and self.keys[name] == key:
                return self.moveStep(name, shift)
        return None

    @staticmethod
    def moveStep(name, shift=False):
        step = MOVE_STEPS[name]
        if step[0] == "nudge":
            factor = 10 if shift else 1
            return ("nudge", step[1] * factor, step[2] * factor)
        angle = step[1]
        if shift and abs(angle) == 5:
            angle *= 18
        return ("turn", angle)

    def actionForMouse(self, button, kinds):
        mouse = mouseName(button)
        if mouse is None:
            return None
        for name in IDS:
            if KIND[name] in kinds and self.mouse[name] == mouse:
                return name
        return None

    # ---- texts ------------------------------------------------------------------------------------------------------------

    def text(self, name):
        """The key of an action as shown to the user ("" if none)."""
        return keyText(self.keys.get(name, 0))

    def mouseText(self, name):
        return MOUSE_TEXT.get(self.mouse.get(name, ""), "")

    def describe(self, name):
        """
        "K", "K / middle button", "middle button" or "" (no key and no button).
        """
        parts = [t for t in (self.text(name), self.mouseText(name)) if t]
        return " / ".join(parts)

    def moveKeysText(self):
        """How the Move Region tool texts name the move keys."""
        names = ["move_left", "move_right", "move_up", "move_down"]
        if all(self.keys[n] == keyCode(DEFAULT_KEY[n]) for n in names):
            return "the arrow keys"
        keys = [self.text(n) for n in names]
        if not any(keys):
            return "the move keys (none set)"
        return "the keys " + " ".join(k if k else "-" for k in keys) + " (left right up down)"

    def turnKeysText(self):
        keys = [k for k in (self.text("turn_left"), self.text("turn_right")) if k]
        return " and ".join(keys) if keys else "the turn keys (none set)"

    # ---- conflicts ---------------------------------------------------------------------------------------------------------

    def conflicts(self):
        """
        Returns (errors, notes, names): errors are shortcuts that can't work (the same key or button for two actions that
        are active together, a reserved key); notes are allowed overlaps (a review key that is also a normal key: the
        review takes it while it is open) and what the right button does; names are the actions in the errors.
        """
        errors = []
        notes = []
        bad = set()
        reserved = {keyCode(k) for k in RESERVED_KEYS}

        def group(name):
            kind = KIND[name]
            return "global" if kind in GLOBAL_KINDS else kind

        def describeKey(code):
            return "'" + keyText(code) + "'"

        # the codes each action occupies: a Move key also takes SHIFT + key (its bigger step)
        occupied = {}
        for name in IDS:
            code = self.keys[name]
            if code == 0:
                continue
            if KIND[name] != FIXED and code in reserved:
                errors.append(describeKey(code) + " can't be used (" + LABEL[name] + "): it is reserved.")
                bad.add(name)
                continue
            codes = [code]
            if KIND[name] == MOVE:
                codes.append(code | int(Qt.ShiftModifier))
            for c in codes:
                occupied.setdefault(c, []).append(name)

        for code, names in occupied.items():
            if len(names) < 2:
                continue
            by_group = {}
            for name in names:
                by_group.setdefault(group(name), []).append(name)
            # the same key twice in one group, or a Move key that is also a review or normal key
            clashing = [n for g, ns in by_group.items() if len(ns) > 1 for n in ns]
            if "move" in by_group and len(by_group) > 1:
                clashing = names
            elif "review" in by_group and "global" in by_group and len(by_group) == 2 and not clashing:
                notes.append(describeKey(code) + ": " + LABEL[by_group["review"][0]] + " while the review is open, " +
                             LABEL[by_group["global"][0]] + " otherwise.")
            if clashing:
                errors.append(describeKey(code) + " is used by: " + ", ".join(LABEL[n] for n in clashing) + ".")
                bad.update(clashing)

        for mouse in MOUSE_NAMES:
            names = [name for name in IDS if self.mouse[name] == mouse]
            if len(names) == 0:
                continue
            by_group = {}
            for name in names:
                by_group.setdefault(group(name), []).append(name)
            clashing = [n for g, ns in by_group.items() if len(ns) > 1 for n in ns]
            if "move" in by_group and len(by_group) > 1:
                clashing = names
            if clashing:
                errors.append("The " + MOUSE_TEXT[mouse] + " is used by: " + ", ".join(LABEL[n] for n in clashing) + ".")
                bad.update(clashing)
                continue
            if "review" in by_group and "global" in by_group:
                notes.append("The " + MOUSE_TEXT[mouse] + ": " + LABEL[by_group["review"][0]] +
                             " while the review is open, " + LABEL[by_group["global"][0]] + " otherwise.")
            if mouse == "Right":
                if "global" in by_group:
                    notes.append("The right button runs '" + LABEL[by_group["global"][0]] + "' instead of opening the "
                                 "region menu (its items are also in the Regions menu).")
                elif "review" in by_group:
                    notes.append("The right button opens the region menu when the review is closed.")
                else:
                    notes.append("The right button opens the region menu when the Move Region tool has no selected "
                                 "region.")
        return errors, notes, bad


# the shortcuts in use (TagLab loads them at start and after File > Keyboard and Mouse Shortcuts)
bindings = Bindings()
